import json
import base64
import asyncio
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from deusauditron.app_logging.logger import logger
from deusauditron.llm_abstraction.llm_helper import LLMInvoker
from deusauditron.schemas.shared_models.models import Message, MessageRole
from deusauditron.config import get_config

import aiohttp
import io
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent


@dataclass
class SpeakerSegment:
    """A segment of audio with speaker identification"""
    speaker: str  # "user" or "assistant"
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class TranscriptData:
    """Parsed transcript data from JSON"""
    user_segments: List[str]
    assistant_segments: List[str]
    full_conversation: List[SpeakerSegment]


@dataclass
class TranscriptionRequest:
    """Request for audio transcription"""
    audio_data: bytes
    audio_format: str
    language: Optional[str] = None
    api_keys: Optional[Dict[str, str]] = None


@dataclass
class InternalTranscriberConfig:
    """Internal configuration for transcription"""
    strategy: str
    model: Optional['TranscriptionModel'] = None
    timeout: int = 60
    aws_region: str = "us-east-1"
    language: str = "en-US"


@dataclass
class TranscriptionModel:
    """Model configuration for transcription"""
    name: str
    provider: str


@dataclass
class TranscriptionResult:
    """Result from transcription"""
    transcript: str
    confidence: Optional[float] = None


class TranscriptionError(Exception):
    """Exception for transcription errors"""
    pass


class ITranscriptionStrategy(ABC):
    """Interface for transcription strategies"""
    
    @abstractmethod
    async def transcribe(self, request: TranscriptionRequest, config: InternalTranscriberConfig) -> TranscriptionResult:
        """Transcribe audio data"""
        pass


class VoiceProcessor:
    """Handles voice processing for STT/TTS evaluation"""

    @staticmethod
    async def convert_audio_to_text(audio_data: bytes, audio_format: str = "ogg", api_keys: Optional[Dict[str, str]] = None) -> str:
        """
        Convert audio data to text using configured transcription strategy
        
        Args:
            audio_data: Raw audio data in bytes
            audio_format: Audio format (ogg, wav, mp3, etc.)
            api_keys: API keys for external services
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Converting audio to text using configured transcription strategy")
            return await TranscriptionService.transcribe_audio(audio_data, audio_format, api_keys)
        except Exception as e:
            logger.error(f"Error converting audio to text: {e}")
            return ""

    @staticmethod
    def parse_transcript_json(transcript_json: str) -> TranscriptData:
        """
        Parse transcript JSON to extract user and assistant segments
        
        Expected format:
        {
            "messages": [
                {
                    "content": "Hello there!",
                    "role": "assistant", 
                    "metadata": {"timestamp": "1756474483.632892"}
                },
                {
                    "content": "Hello.",
                    "role": "user",
                    "metadata": {"timestamp": "1756474486.135797"}
                }
            ]
        }
        
        Args:
            transcript_json: JSON string containing transcript data
            
        Returns:
            TranscriptData with separated user and assistant segments
        """
        try:
            transcript_data = json.loads(transcript_json)
            
            user_segments = []
            assistant_segments = []
            full_conversation = []
            
            # Handle the expected format with "messages" array
            if isinstance(transcript_data, dict) and "messages" in transcript_data:
                messages = transcript_data["messages"]
            elif isinstance(transcript_data, list):
                # Fallback: assume the list itself contains the messages
                messages = transcript_data
            else:
                logger.error(f"Unsupported transcript format. Expected dict with 'messages' key, got: {type(transcript_data)}")
                return TranscriptData([], [], [])
            
            if not isinstance(messages, list):
                logger.error(f"Messages should be a list, got: {type(messages)}")
                return TranscriptData([], [], [])
            
            for message in messages:
                if not isinstance(message, dict):
                    logger.warning(f"Skipping invalid message format: {message}")
                    continue
                
                # Extract role and content from the message
                role = message.get("role", "").lower()
                content_raw = message.get("content", "")
                # Some transcripts may carry content as a list (tokens/segments). Normalize to string.
                if isinstance(content_raw, list):
                    content = " ".join([str(x).strip() for x in content_raw if str(x).strip()])
                else:
                    content = str(content_raw).strip()
                metadata = message.get("metadata", {})
                
                if not content:
                    continue  # Skip empty messages
                
                # Map roles to speaker types
                if role == "user":
                    speaker = "user"
                elif role == "assistant":
                    speaker = "assistant"
                else:
                    logger.warning(f"Unknown role '{role}', skipping message")
                    continue
                
                # Extract timestamp if available
                timestamp = metadata.get("timestamp") if isinstance(metadata, dict) else None
                start_time = float(timestamp) if timestamp and str(timestamp).replace('.', '').isdigit() else None
                
                # Create segment
                segment = SpeakerSegment(
                    speaker=speaker,
                    text=content,
                    start_time=start_time,
                    end_time=None  # End time not provided in this format
                )
                full_conversation.append(segment)
                
                # Add to appropriate list
                if speaker == "user":
                    user_segments.append(content)
                elif speaker == "assistant":
                    assistant_segments.append(content)
            
            logger.info(f"Parsed transcript: {len(user_segments)} user segments, {len(assistant_segments)} assistant segments")
            return TranscriptData(user_segments, assistant_segments, full_conversation)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing transcript JSON: {e}")
            return TranscriptData([], [], [])
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            return TranscriptData([], [], [])


    @staticmethod
    def parse_generated_transcript(generated_text: str) -> TranscriptData:
        """
        Parse LLM-generated transcript text into user and assistant segments
        
        Args:
            generated_text: Text generated by LLM transcription
            
        Returns:
            TranscriptData with separated segments
        """
        user_segments = []
        assistant_segments = []
        full_conversation = []
        
        lines = generated_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for speaker prefixes
            if line.startswith("User:") or line.startswith("Human:") or line.startswith("Customer:"):
                text = line.split(":", 1)[1].strip()
                if text:
                    user_segments.append(text)
                    full_conversation.append(SpeakerSegment("user", text))
                    
            elif line.startswith("Assistant:") or line.startswith("AI:") or line.startswith("Bot:") or line.startswith("Agent:"):
                text = line.split(":", 1)[1].strip()
                if text:
                    assistant_segments.append(text)
                    full_conversation.append(SpeakerSegment("assistant", text))
            # ElevenLabs-style diarized segments: Speaker-<id>: text
            elif line.startswith("Speaker-") and ":" in line:
                text = line.split(":", 1)[1].strip()
                # Heuristic: first speaker we treat as assistant, second as user
                # Accumulate order across lines
                # We'll track seen order in local state
                # Simple approach: if we have fewer assistant segments, assign assistant first
                if len(assistant_segments) <= len(user_segments):
                    if text:
                        assistant_segments.append(text)
                        full_conversation.append(SpeakerSegment("assistant", text))
                else:
                    if text:
                        user_segments.append(text)
                        full_conversation.append(SpeakerSegment("user", text))
            
            # Handle unmarked text - try to infer based on context or position
            elif ":" not in line and line:
                # For now, skip unmarked text or could implement smarter inference
                continue
        
        logger.info(f"Parsed generated transcript: {len(user_segments)} user segments, {len(assistant_segments)} assistant segments")
        return TranscriptData(user_segments, assistant_segments, full_conversation)

    @staticmethod
    def separate_speakers_from_audio_transcript(transcript_text: str) -> Tuple[str, str]:
        """
        Separate user and assistant speech from a transcript
        
        Args:
            transcript_text: Full transcript text
            
        Returns:
            Tuple of (user_text, assistant_text)
        """
        parsed_data = VoiceProcessor.parse_generated_transcript(transcript_text)
        
        user_text = " ".join(parsed_data.user_segments)
        assistant_text = " ".join(parsed_data.assistant_segments)
        
        return user_text.strip(), assistant_text.strip()


class LLMTranscriptionStrategy(ITranscriptionStrategy):
    """Transcription strategy using LLM (via LiteLLM)."""
    
    async def transcribe(self, request: TranscriptionRequest, config: InternalTranscriberConfig) -> TranscriptionResult:
        """Transcribe using LLM API (LiteLLM)."""
        try:
            # Import litellm here to avoid dependency issues if not installed
            try:
                import litellm
            except ImportError:
                raise TranscriptionError("litellm package not installed. Install with: pip install litellm")
            
            audio_file = BytesIO(request.audio_data)
            setattr(audio_file, 'name', f"audio.{request.audio_format}")

            if not config.model:
                raise TranscriptionError("Model configuration is required for LLM transcription")
            
            api_key = self._get_api_key_for_model(config.model.name, request.api_keys or {})
            
            litellm_params = {
                "model": config.model.name,
                "file": audio_file,
                "timeout": config.timeout
            }
            
            if api_key:
                litellm_params["api_key"] = api_key
            
            if request.language:
                litellm_params["language"] = request.language
            
            response = await litellm.atranscription(**litellm_params)
            
            transcript = response.text if hasattr(response, 'text') else str(response)
            
            if not transcript:
                raise TranscriptionError("No transcription text received from LLM provider")
            
            return TranscriptionResult(
                transcript=transcript,
                confidence=None
            )
            
        except Exception as e:
            logger.error(f"LLM transcription failed: {e}")
            raise TranscriptionError(f"LLM transcription failed: {str(e)}")
    
    def _get_api_key_for_model(self, model_name: str, api_keys: dict) -> str:
        """Get the appropriate API key based on the model provider."""
        provider = model_name.split("/")[0].lower()
        config = get_config()
        if provider in config.llm.providers_without_api_key:
            return ""
        
        if provider in api_keys:
            return api_keys[provider]
        
        raise TranscriptionError(f"API key required for provider '{provider}' but not found in api_keys")


class AWSTranscriptionStrategy(ITranscriptionStrategy):
    """Transcription strategy using AWS Transcribe."""
    
    async def transcribe(self, request: TranscriptionRequest, config: InternalTranscriberConfig) -> TranscriptionResult:
        try:
            logger.info(f"AWS Transcription: Starting with format {request.audio_format}")
            if request.audio_format.lower() not in ['pcm', 'wav']:
                raise TranscriptionError(f"AWS Transcribe streaming only supports PCM/WAV format, got: {request.audio_format}")
            
            class TranscriptHandler(TranscriptResultStreamHandler):
                def __init__(self, output_stream):
                    super().__init__(output_stream)
                    self.transcript = ""
                    self.confidence_scores = []
                
                async def handle_transcript_event(self, transcript_event: TranscriptEvent):
                    results = transcript_event.transcript.results
                    if results:
                        for result in results:
                            if not result.is_partial and result.alternatives:
                                for alt in result.alternatives:
                                    self.transcript += alt.transcript + " "
            
            client = TranscribeStreamingClient(region=config.aws_region)
            language_code = request.language if request.language else config.language
            
            logger.info(f"AWS Transcription: Starting stream with language={language_code}")
            stream = await client.start_stream_transcription(
                language_code=language_code,
                media_sample_rate_hz=16000,
                media_encoding="pcm"
            )
            
            handler = TranscriptHandler(stream.output_stream)
            
            async def write_audio():
                audio_stream = io.BytesIO(request.audio_data)
                chunk_size = 1024 * 8
                while True:
                    chunk = audio_stream.read(chunk_size)
                    if not chunk:
                        break
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                    await asyncio.sleep(0.01)
                await stream.input_stream.end_stream()
            
            logger.info("AWS Transcription: Starting audio processing")
            await asyncio.gather(write_audio(), handler.handle_events())
            
            transcript_text = handler.transcript.strip()
            logger.info(f"AWS Transcription: Final transcript length: {len(transcript_text)}")

            if not transcript_text:
                raise TranscriptionError("No transcription result received from AWS")
            
            return TranscriptionResult(
                transcript=transcript_text,
                confidence=0.95
            )
            
        except ImportError:
            raise TranscriptionError("amazon-transcribe package not installed")
        except Exception as e:
            logger.error(f"AWS transcription failed: {e}")
            raise TranscriptionError(f"AWS transcription failed: {str(e)}")


class DeepgramTranscriptionStrategy(ITranscriptionStrategy):
    """Strategy for Deepgram-based transcription."""
    
    async def transcribe(self, request: TranscriptionRequest, config: InternalTranscriberConfig) -> TranscriptionResult:
        """Transcribe using Deepgram API."""
        try:
            if not request.api_keys or "deepgram" not in request.api_keys:
                raise TranscriptionError("Deepgram API key required but not found in api_keys")
                
            api_key = request.api_keys["deepgram"]
            app_config = get_config()
            
            headers = {
                "Authorization": f"Token {api_key}",
                "Content-Type": f"audio/{request.audio_format}"
            }
            
            model_name = (
                config.model.name if (config.model and getattr(config.model, "name", None)) else app_config.deepgram.model
            )
            params = {
                "model": model_name,
                "smart_format": str(app_config.deepgram.smart_format).lower(),
                "punctuate": str(app_config.deepgram.punctuate).lower(),
            }

            if request.language:
                params["language"] = request.language
            
            timeout = aiohttp.ClientTimeout(total=config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # logger.info(f"Deepgram API data: {request.audio_data}")
                async with session.post(
                    app_config.deepgram.api_url,
                    headers=headers,
                    params=params,
                    data=request.audio_data
                ) as response:
                    logger.info(f"Deepgram API URL: {app_config.deepgram.api_url}")
                    logger.info(f"Deepgram API headers: {headers}")
                    logger.info(f"Deepgram API params: {params}")
                    logger.info(f"Deepgram API response: {response}")
                    if response.status != 200:
                        error_text = await response.text()
                        raise TranscriptionError(f"Deepgram API error ({response.status}): {error_text}")
                    
                    result = await response.json()
                    
                    transcript = self._extract_deepgram_transcript(result)
                    confidence = self._extract_deepgram_confidence(result)
                    return TranscriptionResult(
                        transcript=transcript,
                        confidence=confidence
                    )
                    
        except Exception as e:
            raise TranscriptionError(f"Deepgram transcription failed: {str(e)}")
    
    def _extract_deepgram_transcript(self, result: dict) -> str:
        """Extract transcript from Deepgram response."""
        try:
            alternatives = result["results"]["channels"][0]["alternatives"]
            if not alternatives:
                raise TranscriptionError("No transcript alternatives found")
            
            transcript = alternatives[0]["transcript"].strip()
            if not transcript:
                raise TranscriptionError("Empty transcript received")
            
            return transcript
        except (KeyError, IndexError) as e:
            raise TranscriptionError(f"Invalid Deepgram response format: {str(e)}")

    def _extract_deepgram_confidence(self, result: dict) -> Optional[float]:
        """Extract confidence score from Deepgram response."""
        try:
            alternatives = result["results"]["channels"][0]["alternatives"]
            if alternatives and "confidence" in alternatives[0]:
                return alternatives[0]["confidence"]
            return None
        except (KeyError, IndexError):
            return None


class ElevenLabsTranscriptionStrategy(ITranscriptionStrategy):
    """Strategy for ElevenLabs-based transcription with diarization."""
    
    async def transcribe(self, request: TranscriptionRequest, config: InternalTranscriberConfig) -> TranscriptionResult:
        try:
            if not request.api_keys or "elevenlabs" not in request.api_keys:
                raise TranscriptionError("ElevenLabs API key required but not found in api_keys")
            api_key = request.api_keys["elevenlabs"]
            app_config = get_config()
            
            # Upload audio for transcription (create job)
            # ElevenLabs expects multipart form with file + model_id (and optional fields)
            headers = {
                "xi-api-key": api_key,
            }
            form = aiohttp.FormData()
            filename = f"audio.{request.audio_format}"
            form.add_field(
                name="file",
                value=request.audio_data,
                filename=filename,
                content_type=f"audio/{request.audio_format}",
            )
            form.add_field("model_id", app_config.elevenlabs.model)
            form.add_field("diarize", str(app_config.elevenlabs.diarize).lower())
            # ElevenLabs uses language_code
            # form.add_field("language_code", (request.language or config.language))
            timeout = aiohttp.ClientTimeout(total=config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    app_config.elevenlabs.upload_url,
                    headers=headers,
                    data=form,
                ) as create_resp:
                    if create_resp.status not in (200, 201, 202):
                        error_text = await create_resp.text()
                        raise TranscriptionError(f"ElevenLabs create transcript error ({create_resp.status}): {error_text}")
                    create_json = await create_resp.json()
                    transcription_id = create_json.get("transcription_id") or create_json.get("id")
                    if not transcription_id:
                        raise TranscriptionError("ElevenLabs response missing transcription_id")

                # Poll transcript until ready with retry mechanism
                transcript_url = f"{app_config.elevenlabs.transcripts_url}/{transcription_id}"
                max_retries = app_config.elevenlabs.max_poll_retries
                poll_interval = app_config.elevenlabs.poll_interval
                
                for attempt in range(max_retries):
                    try:
                        async with session.get(transcript_url, headers={"xi-api-key": api_key}) as get_resp:
                            if get_resp.status != 200:
                                error_text = await get_resp.text()
                                logger.warning(f"ElevenLabs get transcript attempt {attempt + 1}/{max_retries} failed ({get_resp.status}): {error_text}")
                                if attempt == max_retries - 1:
                                    raise TranscriptionError(f"ElevenLabs get transcript error ({get_resp.status}): {error_text}")
                            else:
                                result = await get_resp.json()
                                # Simple readiness check: presence of "text" or non-empty "words"
                                if result.get("text") or result.get("words"):
                                    transcript, confidence = self._extract_transcript_and_confidence(result)
                                    logger.info(f"ElevenLabs transcript ready after {attempt + 1} attempts")
                                    return TranscriptionResult(transcript=transcript, confidence=confidence)
                    except asyncio.TimeoutError:
                        logger.warning(f"ElevenLabs polling timeout on attempt {attempt + 1}/{max_retries}")
                        if attempt == max_retries - 1:
                            raise TranscriptionError("ElevenLabs transcription polling timeout")
                    
                    await asyncio.sleep(poll_interval)
                
                raise TranscriptionError(f"ElevenLabs transcript not ready after {max_retries} attempts")
        except Exception as e:
            raise TranscriptionError(f"ElevenLabs transcription failed: {str(e)}")

    def _extract_transcript_and_confidence(self, result: dict) -> Tuple[str, Optional[float]]:
        # Prefer words array with diarization for assembling transcript with speaker markers
        words = result.get("words") or []
        if words:
            # Build lines like "Speaker-<id>: <joined words>" preserving order
            speaker_to_chunks: Dict[str, List[str]] = {}
            for w in words:
                speaker_id = str(w.get("speaker_id", "0"))
                text = w.get("text", "").strip()
                if not text:
                    continue
                speaker_to_chunks.setdefault(speaker_id, []).append(text)
            # Map first encountered speaker to assistant, second to user for our use-case
            ordered_speakers = list(speaker_to_chunks.keys())
            lines: List[str] = []
            for spk in ordered_speakers:
                line = f"Speaker-{spk}: {' '.join(speaker_to_chunks[spk])}".strip()
                lines.append(line)
            transcript_text = "\n".join(lines)
            return transcript_text, None
        # Fallback to flat text
        return result.get("text", ""), None

class TranscriptionStrategyFactory:
    """Factory for creating transcription strategies."""
    
    @staticmethod
    def create_strategy(strategy_name: str) -> ITranscriptionStrategy:
        """Create a transcription strategy based on the strategy name."""
        if strategy_name.lower() == "llm":
            return LLMTranscriptionStrategy()
        elif strategy_name.lower() == "aws":
            return AWSTranscriptionStrategy()
        elif strategy_name.lower() == "deepgram":
            return DeepgramTranscriptionStrategy()
        elif strategy_name.lower() == "elevenlabs":
            return ElevenLabsTranscriptionStrategy()
        else:
            raise TranscriptionError(f"Unknown transcription strategy: {strategy_name}")


class TranscriptionService:
    """Service for handling audio transcription with different strategies."""
    
    @staticmethod
    async def transcribe_audio(audio_data: bytes, audio_format: str = "ogg", api_keys: Optional[Dict[str, str]] = None) -> str:
        """
        Transcribe audio using configured strategy.
        
        Args:
            audio_data: Raw audio data in bytes
            audio_format: Audio format (ogg, wav, mp3, etc.)
            api_keys: API keys for external services
            
        Returns:
            Transcribed text
        """
        try:
            config = get_config()
            transcription_config = config.transcription
            
            # Create internal config
            internal_config = InternalTranscriberConfig(
                strategy=transcription_config.strategy,
                model=TranscriptionModel(
                    name=transcription_config.llm_model,
                    provider=transcription_config.llm_model.split("/")[0] if "/" in transcription_config.llm_model else "openai"
                ),
                timeout=transcription_config.timeout,
                aws_region=transcription_config.aws_region,
                language=transcription_config.language
            )
            
            # Create request
            request = TranscriptionRequest(
                audio_data=audio_data,
                audio_format=audio_format,
                language=transcription_config.language,
                api_keys=api_keys
            )
            
            # Get strategy and transcribe
            strategy = TranscriptionStrategyFactory.create_strategy(transcription_config.strategy)
            result = await strategy.transcribe(request, internal_config)
            
            logger.info(f"Transcription completed using {transcription_config.strategy} strategy")
            return result.transcript
            
        except Exception as e:
            logger.error(f"Transcription service failed: {e}")
            raise TranscriptionError(f"Transcription service failed: {str(e)}")
