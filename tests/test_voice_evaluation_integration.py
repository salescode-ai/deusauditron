import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from deusauditron.schemas.shared_models.models import VoiceEvaluationPayload, VoiceEvalRequest
from deusauditron.schemas.autogen.references.voice_evaluation_result_schema import (
    VoiceEvaluationResult, VoiceEvalStatus, VoiceEvalCategory
)
from deusauditron.voice.voice_utils import VoiceProcessor, TranscriptData, SpeakerSegment
from deusauditron.eval.voice_eval_strategies import STTEvaluationStrategy, TTSEvaluationStrategy


class TestVoiceEvaluationIntegration:
    """Integration tests for voice evaluation functionality"""

    def test_voice_evaluation_payload_creation(self):
        """Test creating a voice evaluation payload"""
        payload = VoiceEvaluationPayload(
            api_keys={"openai": "test-key"},
            recording_path="s3://test-bucket/recording.ogg",
            transcript_path="s3://test-bucket/transcript.json",
            voice_eval_result="test-bucket/results/"
        )
        
        assert payload.recording_path == "s3://test-bucket/recording.ogg"
        assert payload.transcript_path == "s3://test-bucket/transcript.json"
        assert payload.voice_eval_result == "test-bucket/results/"
        assert payload.api_keys["openai"] == "test-key"

    def test_voice_eval_request_creation(self):
        """Test creating a voice evaluation request"""
        payload = VoiceEvaluationPayload(
            recording_path="s3://test-bucket/recording.ogg",
            transcript_path="s3://test-bucket/transcript.json",
            voice_eval_result="test-bucket/results/"
        )
        
        request = VoiceEvalRequest(
            tenant_id="test-tenant",
            agent_id="test-agent",
            run_id="test-run",
            payload=payload
        )
        
        assert request.tenant_id == "test-tenant"
        assert request.agent_id == "test-agent"
        assert request.run_id == "test-run"
        assert request.composite_key == "test-tenant:test-agent:test-run"

    def test_voice_evaluation_result_creation(self):
        """Test creating a voice evaluation result"""
        result = VoiceEvaluationResult(
            status=VoiceEvalStatus.Completed,
            progress=100,
            overall_accuracy=85.5
        )
        
        assert result.status == VoiceEvalStatus.Completed
        assert result.progress == 100
        assert result.overall_accuracy == 85.5

    def test_transcript_json_parsing(self):
        """Test parsing the actual transcript JSON format"""
        # Test the actual format used in production
        transcript_json = json.dumps({
            "messages": [
                {
                    "content": "Hello, how are you?",
                    "role": "user",
                    "metadata": {"timestamp": "1756474483.632892"}
                },
                {
                    "content": "I'm doing well, thank you!",
                    "role": "assistant", 
                    "metadata": {
                        "intent": "greeting",
                        "intent_detector": "introduction",
                        "timestamp": "1756474486.135797"
                    }
                },
                {
                    "content": "What's the weather like?",
                    "role": "user",
                    "metadata": {"timestamp": "1756474490.123456"}
                },
                {
                    "content": "It's sunny and warm today.",
                    "role": "assistant",
                    "metadata": {
                        "intent": "weather",
                        "node": "weather_response",
                        "timestamp": "1756474495.789012"
                    }
                }
            ]
        })
        
        result = VoiceProcessor.parse_transcript_json(transcript_json)
        assert len(result.user_segments) == 2
        assert len(result.assistant_segments) == 2
        assert result.user_segments[0] == "Hello, how are you?"
        assert result.user_segments[1] == "What's the weather like?"
        assert result.assistant_segments[0] == "I'm doing well, thank you!"
        assert result.assistant_segments[1] == "It's sunny and warm today."
        
        # Test timestamp extraction
        assert len(result.full_conversation) == 4
        assert result.full_conversation[0].start_time == 1756474483.632892
        assert result.full_conversation[1].start_time == 1756474486.135797

    def test_generated_transcript_parsing(self):
        """Test parsing LLM-generated transcript text"""
        generated_text = """
        User: Hello, I need help with my account
        Assistant: I'd be happy to help you with your account. What specific issue are you experiencing?
        User: I can't log in
        Assistant: Let me help you troubleshoot that login issue.
        """
        
        result = VoiceProcessor.parse_generated_transcript(generated_text)
        assert len(result.user_segments) == 2
        assert len(result.assistant_segments) == 2
        assert "Hello, I need help with my account" in result.user_segments[0]
        assert "I can't log in" in result.user_segments[1]

    @pytest.mark.asyncio
    async def test_stt_evaluation_strategy(self):
        """Test STT evaluation strategy"""
        strategy = STTEvaluationStrategy({"openai": "test-key"})
        
        generated_text = "Hello how are you today"
        actual_text = "Hello, how are you today?"
        
        with patch.object(strategy, '_get_llm_analysis', return_value="Good transcription quality"):
            result = await strategy.evaluate(generated_text, actual_text)
        
        assert result.accuracy_score > 80  # Should be high for similar texts
        assert result.similarity_score > 80
        assert result.word_error_rate < 20
        assert len(result.identified_flaws) >= 0

    @pytest.mark.asyncio
    async def test_tts_evaluation_strategy(self):
        """Test TTS evaluation strategy"""
        strategy = TTSEvaluationStrategy({"openai": "test-key"})
        
        generated_text = "I can help you with that request"
        actual_text = "I can help you with that request"
        
        with patch.object(strategy, '_get_llm_analysis', return_value="Perfect speech fidelity"):
            result = await strategy.evaluate(generated_text, actual_text)
        
        assert result.accuracy_score > 95  # Should be very high for identical texts
        assert result.similarity_score > 95
        assert result.word_error_rate < 5

    def test_word_error_rate_calculation(self):
        """Test WER calculation accuracy"""
        strategy = STTEvaluationStrategy({})
        
        # Perfect match
        wer = strategy._calculate_word_error_rate("hello world", "hello world")
        assert wer == 0.0
        
        # Complete mismatch
        wer = strategy._calculate_word_error_rate("foo bar", "hello world")
        assert wer == 200.0  # All words wrong + extra insertions
        
        # Partial match
        wer = strategy._calculate_word_error_rate("hello there", "hello world")
        assert 0 < wer < 100

    def test_similarity_calculation(self):
        """Test similarity score calculation"""
        strategy = STTEvaluationStrategy({})
        
        # Identical texts
        similarity = strategy._calculate_similarity_score("hello world", "hello world")
        assert similarity == 100.0
        
        # Similar texts
        similarity = strategy._calculate_similarity_score("hello world", "hello earth")
        assert 50 < similarity < 100
        
        # Very different texts
        similarity = strategy._calculate_similarity_score("hello world", "goodbye universe")
        assert similarity < 50

    def test_text_normalization(self):
        """Test text normalization function"""
        strategy = STTEvaluationStrategy({})
        
        # Test case normalization and punctuation removal
        normalized = strategy._normalize_text("Hello, World! How are you?")
        assert normalized == "hello world how are you"
        
        # Test whitespace normalization
        normalized = strategy._normalize_text("  hello    world  ")
        assert normalized == "hello world"

    def test_speaker_segment_creation(self):
        """Test SpeakerSegment data structure"""
        segment = SpeakerSegment(
            speaker="user",
            text="Hello there",
            start_time=0.0,
            end_time=2.5
        )
        
        assert segment.speaker == "user"
        assert segment.text == "Hello there"
        assert segment.start_time == 0.0
        assert segment.end_time == 2.5

    def test_transcript_data_creation(self):
        """Test TranscriptData data structure"""
        segments = [
            SpeakerSegment("user", "Hello"),
            SpeakerSegment("assistant", "Hi there")
        ]
        
        transcript_data = TranscriptData(
            user_segments=["Hello"],
            assistant_segments=["Hi there"],
            full_conversation=segments
        )
        
        assert len(transcript_data.user_segments) == 1
        assert len(transcript_data.assistant_segments) == 1
        assert len(transcript_data.full_conversation) == 2

    @pytest.mark.asyncio 
    async def test_voice_processor_error_handling(self):
        """Test error handling in voice processor"""
        # Test invalid JSON
        result = VoiceProcessor.parse_transcript_json("invalid json")
        assert len(result.user_segments) == 0
        assert len(result.assistant_segments) == 0
        
        # Test empty transcript
        result = VoiceProcessor.parse_generated_transcript("")
        assert len(result.user_segments) == 0
        assert len(result.assistant_segments) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
