import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, cast

from deusauditron.app_logging.context import set_logging_context
from deusauditron.app_logging.logger import logger
from deusauditron.state.manager import StateManager
from deusauditron.util.helper import Trinity
from deusauditron.voice.voice_utils import VoiceProcessor, TranscriptData
from deusauditron.schemas.autogen.references.voice_evaluation_result_schema import (
    VoiceEvaluationResult, VoiceEvalStatus, VoiceEvaluationDetails, 
    VoiceComparisonResult, VoiceEvalCategory
)
from deusauditron.schemas.shared_models.models import VoiceEvalRequest

from .voice_eval_strategies import STTEvaluationStrategy, TTSEvaluationStrategy


class VoiceEvaluationError(Exception):
    pass


class VoiceEvaluator:
    def __init__(self, request: VoiceEvalRequest):
        self.request = request
        self.tenant_id = request.tenant_id
        self.agent_id = request.agent_id
        self.run_id = request.run_id
        self.payload = request.payload
        self.start_time: Optional[datetime] = None
        set_logging_context(self.tenant_id, self.agent_id, self.run_id)

    async def evaluate(self) -> None:
        """Main evaluation workflow"""
        try:
            logger.info("Starting voice evaluation process")
            logger.info(f"Request: {self.request}")
            await self._update_status(VoiceEvalStatus.In_Progress, 10)
            
            recording_data, transcript_json = await self._load_data_from_s3()
            logger.info(f"Successfully loaded data from S3")
            await self._update_status(VoiceEvalStatus.In_Progress, 25)
            
            generated_transcript, actual_transcript = await self._process_audio_and_transcript(
                recording_data, transcript_json
            )
            logger.info(f"Successfully processed audio and transcript Generated: {generated_transcript} Actual: {actual_transcript}")
            await self._update_status(VoiceEvalStatus.In_Progress, 50)
            
            stt_result, tts_result = await self._execute_evaluations_parallel(
                generated_transcript, actual_transcript
            )
            logger.info(f"Successfully executed evaluations STT: {stt_result} TTS: {tts_result}")
            await self._update_status(VoiceEvalStatus.In_Progress, 85)
    
            await self._finalize_results(stt_result, tts_result)
            logger.info(f"Successfully finalized results Final STT: {stt_result} Final TTS: {tts_result}")
            await self._update_status(VoiceEvalStatus.Completed, 100)
            
            logger.info("Voice evaluation process completed successfully")
            
        except Exception as e:
            logger.error(f"Voice evaluation failed: {e}")
            await self._update_error_state(str(e))
            raise VoiceEvaluationError(f"Voice evaluation failed: {str(e)}") from e

    async def _load_data_from_s3(self) -> tuple[bytes, str]:
        """Load recording and transcript data from S3"""
        if not self.payload:
            raise VoiceEvaluationError("Voice evaluation payload is required")
            
        try:
            logger.info(f"Loading recording from: {self.payload.recording_path}")
            logger.info(f"Loading transcript from: {self.payload.transcript_path}")
            
            # Load recording and transcript in parallel
            recording_task = Trinity.aread_from_s3(self.payload.recording_path, read_as_text=False)
            transcript_task = Trinity.aread_from_s3(self.payload.transcript_path, read_as_text=True)
            
            recording_result, transcript_json = await asyncio.gather(recording_task, transcript_task)
            
            # Convert recording result to bytes if it's a string
            if isinstance(recording_result, str):
                recording_data = recording_result.encode('utf-8')
            elif isinstance(recording_result, bytes):
                recording_data = recording_result
            else:
                raise VoiceEvaluationError("Recording data must be string or bytes")
                
            # Ensure transcript is a string
            if not isinstance(transcript_json, str):
                raise VoiceEvaluationError("Transcript data must be a string")
            
            if not recording_data:
                raise VoiceEvaluationError(f"Failed to load recording from {self.payload.recording_path}")
            if not transcript_json:
                raise VoiceEvaluationError(f"Failed to load transcript from {self.payload.transcript_path}")
            
            # Validate recording data
            if len(recording_data) < 100:  # Minimum reasonable audio file size (100 bytes)
                raise VoiceEvaluationError(f"Recording data too small ({len(recording_data)} bytes) - may be corrupted")
            
            # Validate transcript JSON
            try:
                import json as _json
                parsed_transcript = _json.loads(transcript_json)
                if not isinstance(parsed_transcript, dict):
                    raise VoiceEvaluationError("Transcript must be a JSON object")
                if "messages" not in parsed_transcript:
                    raise VoiceEvaluationError("Transcript JSON must contain 'messages' field")
                if not isinstance(parsed_transcript.get("messages"), list):
                    raise VoiceEvaluationError("Transcript 'messages' must be an array")
                if len(parsed_transcript.get("messages", [])) == 0:
                    raise VoiceEvaluationError("Transcript contains no messages")
            except _json.JSONDecodeError as e:
                raise VoiceEvaluationError(f"Invalid transcript JSON format: {str(e)}")
            
            logger.info(f"Loaded recording: {len(recording_data)} bytes")
            logger.info(f"Loaded transcript: {len(transcript_json)} characters with {len(parsed_transcript.get('messages', []))} messages")
            
            return recording_data, transcript_json
            
        except Exception as e:
            logger.error(f"Error loading data from S3: {e}")
            raise VoiceEvaluationError(f"Failed to load data from S3: {str(e)}") from e

    async def _process_audio_and_transcript(self, recording_data: bytes, transcript_json: str) -> tuple[TranscriptData, TranscriptData]:
        """
        Process audio to generate transcript and parse actual transcript
        
        Expected transcript format:
        {
            "messages": [
                {
                    "content": "Hello there!",
                    "role": "assistant",
                    "metadata": {"timestamp": "1756474483.632892", ...}
                },
                {
                    "content": "Hello.",
                    "role": "user", 
                    "metadata": {"timestamp": "1756474486.135797", ...}
                }
            ]
        }
        """
        try:
            # Convert audio to text using LLM
            logger.info("Converting audio recording to text")
            api_keys = self.payload.api_keys if self.payload else {}
            
            # Add timeout for transcription
            try:
                generated_text = await asyncio.wait_for(
                    VoiceProcessor.convert_audio_to_text(
                        recording_data, 
                        audio_format="ogg",
                        api_keys=api_keys
                    ),
                    timeout=120.0  # 2 minutes timeout for transcription
                )
            except asyncio.TimeoutError:
                raise VoiceEvaluationError("Audio transcription timed out after 120 seconds")
            
            if not generated_text or not generated_text.strip():
                raise VoiceEvaluationError("Failed to convert audio to text - empty transcription returned")
            
            logger.info(f"Generated transcript: {len(generated_text)} characters")
            logger.info(f"Generated transcript: {generated_text}")
            
            # Parse both transcripts
            generated_transcript = VoiceProcessor.parse_generated_transcript(generated_text)
            actual_transcript = VoiceProcessor.parse_transcript_json(transcript_json)
            
            logger.info(f"Generated - User: {len(generated_transcript.user_segments)} segments, "
                       f"Assistant: {len(generated_transcript.assistant_segments)} segments")
            logger.info(f"Actual - User: {len(actual_transcript.user_segments)} segments, "
                       f"Assistant: {len(actual_transcript.assistant_segments)} segments")
            
            return generated_transcript, actual_transcript
            
        except Exception as e:
            logger.error(f"Error processing audio and transcript: {e}")
            raise VoiceEvaluationError(f"Failed to process audio and transcript: {str(e)}") from e

    async def _execute_evaluations_parallel(self, generated_transcript: TranscriptData, actual_transcript: TranscriptData) -> tuple[VoiceEvaluationDetails, VoiceEvaluationDetails]:
        """Execute STT and TTS evaluations in parallel"""
        try:
            logger.info("Executing STT and TTS evaluations in parallel")
            
            # Get API keys for evaluation
            api_keys = self.payload.api_keys if self.payload else {}
            
            # Create evaluation tasks
            stt_task = self._evaluate_stt(generated_transcript, actual_transcript, api_keys)
            tts_task = self._evaluate_tts(generated_transcript, actual_transcript, api_keys)
            
            # Execute in parallel without return_exceptions to get proper typing
            try:
                stt_result, tts_result = await asyncio.gather(stt_task, tts_task)
                return stt_result, tts_result
            except Exception as e:
                # If any task fails, both will be cancelled
                logger.error(f"Parallel evaluation failed: {e}")
                raise VoiceEvaluationError(f"Evaluation failed: {str(e)}") from e
            
        except Exception as e:
            logger.error(f"Error executing parallel evaluations: {e}")
            raise VoiceEvaluationError(f"Failed to execute evaluations: {str(e)}") from e

    async def _evaluate_stt(self, generated_transcript: TranscriptData, actual_transcript: TranscriptData, api_keys: dict) -> VoiceEvaluationDetails:
        """Evaluate Speech-to-Text accuracy"""
        start_time = datetime.now()
        
        try:
            # Extract user text (what user actually said vs what STT detected)
            generated_user_text = " ".join(generated_transcript.user_segments)
            actual_user_text = " ".join(actual_transcript.user_segments)
            
            logger.info(f"STT Evaluation - Generated: {len(generated_user_text)} chars, Actual: {len(actual_user_text)} chars")
            
            # Get evaluation model from payload if provided
            evaluation_model = self.payload.evaluation_model if self.payload else None
            
            # Use STT evaluation strategy
            strategy = STTEvaluationStrategy(api_keys, evaluation_model=evaluation_model)
            # Compute cleaned texts used for comparison (and return them to result)
            try:
                generated_user_text_clean = strategy._clean_transcript_for_stt(generated_user_text)  # type: ignore[attr-defined]
            except Exception:
                generated_user_text_clean = generated_user_text
            try:
                actual_user_text_clean = strategy._clean_transcript_for_stt(actual_user_text)  # type: ignore[attr-defined]
            except Exception:
                actual_user_text_clean = actual_user_text
            comparison_result = await strategy.evaluate(generated_user_text, actual_user_text)
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return VoiceEvaluationDetails(
                category=VoiceEvalCategory.STT,
                generated_text=generated_user_text,
                generated_text_clean=generated_user_text_clean,
                actual_text=actual_user_text,
                actual_text_clean=actual_user_text_clean,
                comparison_result=comparison_result,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"STT evaluation error: {e}")
            raise

    async def _evaluate_tts(self, generated_transcript: TranscriptData, actual_transcript: TranscriptData, api_keys: dict) -> VoiceEvaluationDetails:
        """Evaluate Text-to-Speech accuracy"""
        start_time = datetime.now()
        
        try:
            # Extract assistant text (what AI said vs what was in transcript)
            generated_assistant_text = " ".join(generated_transcript.assistant_segments)
            actual_assistant_text = " ".join(actual_transcript.assistant_segments)
            
            logger.info(f"TTS Evaluation - Generated: {len(generated_assistant_text)} chars, Actual: {len(actual_assistant_text)} chars")
            
            # Get evaluation model from payload if provided
            evaluation_model = self.payload.evaluation_model if self.payload else None
            
            # Use TTS evaluation strategy
            strategy = TTSEvaluationStrategy(api_keys, evaluation_model=evaluation_model)
            # Compute cleaned texts used for comparison (and return them to result)
            try:
                generated_assistant_text_clean = strategy._clean_transcript_for_stt(generated_assistant_text)  # type: ignore[attr-defined]
            except Exception:
                generated_assistant_text_clean = generated_assistant_text
            try:
                actual_assistant_text_clean = strategy._clean_transcript_for_stt(actual_assistant_text)  # type: ignore[attr-defined]
            except Exception:
                actual_assistant_text_clean = actual_assistant_text
            comparison_result = await strategy.evaluate(generated_assistant_text, actual_assistant_text)
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return VoiceEvaluationDetails(
                category=VoiceEvalCategory.TTS,
                generated_text=generated_assistant_text,
                generated_text_clean=generated_assistant_text_clean,
                actual_text=actual_assistant_text,
                actual_text_clean=actual_assistant_text_clean,
                comparison_result=comparison_result,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"TTS evaluation error: {e}")
            raise

    async def _finalize_results(self, stt_result: VoiceEvaluationDetails, tts_result: VoiceEvaluationDetails) -> None:
        """Calculate overall results and save to S3"""
        try:
            # Calculate overall accuracy
            overall_accuracy = (stt_result.comparison_result.accuracy_score + tts_result.comparison_result.accuracy_score) / 2
            
            # Update final state
            voice_eval_state = await StateManager().get_voice_eval_state(self.tenant_id, self.agent_id, self.run_id)
            if not voice_eval_state or not voice_eval_state.voice_evaluation_result:
                logger.error("Voice evaluation state not found - cannot finalize results")
                raise VoiceEvaluationError("Voice evaluation state not found")
                
            voice_eval_state.voice_evaluation_result.stt_evaluation = stt_result
            voice_eval_state.voice_evaluation_result.tts_evaluation = tts_result
            voice_eval_state.voice_evaluation_result.overall_accuracy = overall_accuracy
            voice_eval_state.voice_evaluation_result.end_time = datetime.now(timezone.utc)
            
            await StateManager().set_voice_eval_state(self.tenant_id, self.agent_id, self.run_id, voice_eval_state)
            
            # Save to S3
            await self._save_results_to_s3(voice_eval_state.voice_evaluation_result)
            
            logger.info(f"Voice evaluation completed - Overall accuracy: {overall_accuracy:.2f}%")
            
        except Exception as e:
            logger.error(f"Error finalizing results: {e}")
            raise

    async def _save_results_to_s3(self, result: VoiceEvaluationResult) -> None:
        """Save evaluation results to S3"""
        try:
            if not self.payload or not self.payload.voice_eval_result:
                logger.warning("No S3 path provided for saving results")
                return
                
            # Support either raw "bucket/path" or full "s3://bucket/path" in payload
            persist_path = self.payload.voice_eval_result
            if persist_path.startswith("s3://"):
                # Strip scheme
                persist_path = persist_path[len("s3://"):]
            bucket_name = persist_path.split("/")[0]
            object_key = "/".join(persist_path.split("/")[1:])
            
            # Ensure it ends with voice_eval_result.json
            if not object_key.endswith("voice_eval_result.json"):
                if object_key.endswith("/"):
                    object_key += "voice_eval_result.json"
                else:
                    object_key += "/voice_eval_result.json"
            
            # Convert result to JSON
            result_dict = result.model_dump()
            json_data = json.dumps(result_dict, indent=2, default=str)
            
            try:
                Trinity.write_to_s3(bucket_name, object_key, json_data)
                logger.info(f"Successfully saved voice evaluation results to S3: s3://{bucket_name}/{object_key}")
            except Exception as write_error:
                logger.error(f"Failed to write voice evaluation results to S3: {write_error}")
                raise VoiceEvaluationError(f"Failed to save results to S3: {str(write_error)}")
                
        except Exception as e:
            logger.error(f"Error saving results to S3: {e}")

    async def _update_status(self, status: VoiceEvalStatus, progress: int) -> None:
        """Update evaluation status and progress"""
        try:
            voice_eval_state = await StateManager().get_voice_eval_state(self.tenant_id, self.agent_id, self.run_id)
            if voice_eval_state and voice_eval_state.voice_evaluation_result:
                voice_eval_state.voice_evaluation_result.status = status
                voice_eval_state.voice_evaluation_result.progress = progress
                
                if status == VoiceEvalStatus.In_Progress and not voice_eval_state.voice_evaluation_result.start_time:
                    voice_eval_state.voice_evaluation_result.start_time = datetime.now(timezone.utc)
                
                await StateManager().set_voice_eval_state(self.tenant_id, self.agent_id, self.run_id, voice_eval_state)
                
        except Exception as e:
            logger.error(f"Error updating status: {e}")

    async def _update_error_state(self, error_message: str) -> None:
        """Update state to error status"""
        try:
            voice_eval_state = await StateManager().get_voice_eval_state(self.tenant_id, self.agent_id, self.run_id)
            if voice_eval_state and voice_eval_state.voice_evaluation_result:
                voice_eval_state.voice_evaluation_result.status = VoiceEvalStatus.Error
                voice_eval_state.voice_evaluation_result.error_message = error_message
                voice_eval_state.voice_evaluation_result.end_time = datetime.now(timezone.utc)
                
                await StateManager().set_voice_eval_state(self.tenant_id, self.agent_id, self.run_id, voice_eval_state)
                
        except Exception as e:
            logger.error(f"Error updating error state: {e}")
