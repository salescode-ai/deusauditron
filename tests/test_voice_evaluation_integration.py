import os
import pytest
import pytest_asyncio
from httpx import AsyncClient
from fastapi import status

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

SKIP_LIVE_TESTS = not all([OPENAI_API_KEY, DEEPGRAM_API_KEY])
SKIP_MESSAGE = """
E2E voice evaluation tests require API keys and a running server. 

SETUP:
1. Start the server: uvicorn deusauditron.app:app --host 0.0.0.0 --port 8081
2. Set environment variables:
  - OPENAI_API_KEY: Your OpenAI API key (required)
  - DEEPGRAM_API_KEY: Your Deepgram API key (required)
  - GEMINI_API_KEY: Your Gemini API key (optional)
  - ELEVENLABS_API_KEY: Your ElevenLabs API key (optional)
"""

BASE_URL = "http://localhost:8081"


class TestVoiceEvaluationRouter:

    @pytest_asyncio.fixture
    async def client(self):
        async with AsyncClient(base_url=BASE_URL, timeout=30.0) as ac:
            yield ac

    @pytest.fixture
    def test_ids(self):
        return {
            "tenant_id": "hccbckinddemo",
            "agent_id": "english",
            "run_id": "RM_pPWcrz9vAsLT",
        }

    @pytest.fixture
    def valid_voice_eval_payload(self):
        return {
            "voice_eval_result": "s3://livekit-agent-mgtind-dev/scai/hccbckinddemo/english/300925/RM_pPWcrz9vAsLT/recording/voice_eval_result.json",
            "transcript_path": "s3://livekit-agent-mgtind-dev/scai/hccbckinddemo/english/300925/RM_pPWcrz9vAsLT/recording/transcripts.json",
            "recording_path": "s3://livekit-agent-mgtind-dev/scai/hccbckinddemo/english/300925/RM_pPWcrz9vAsLT/recording/sip_room_49356ae0-2a22-4378-b9ad-e5507f8741c9_30-09-25 09:49.ogg",
            "api_keys": {
                "openai": OPENAI_API_KEY or "",
                "deepgram": DEEPGRAM_API_KEY or "",
                "gemini": GEMINI_API_KEY or "",
                "elevenlabs": ELEVENLABS_API_KEY or "",
            },
        }

    @pytest.mark.asyncio
    async def test_post_voice_evaluation_without_state_creates_new(
        self, client, test_ids, valid_voice_eval_payload
    ):
        url = f"/internal/api/v1/agents/{test_ids['tenant_id']}/{test_ids['agent_id']}/{test_ids['run_id']}/evaluation/voice"
        
        response = await client.post(url, json=valid_voice_eval_payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] in ["Requested", "InProgress", "Completed", "Failed"]
        assert "progress" in data

    @pytest.mark.asyncio
    async def test_post_voice_evaluation_with_invalid_s3_paths(
        self, client, test_ids
    ):
        url = f"/internal/api/v1/agents/{test_ids['tenant_id']}/{test_ids['agent_id']}/{test_ids['run_id']}/evaluation/voice"
        
        payload = {
            "voice_eval_result": "s3://bucket/path/result.json",
            "transcript_path": "s3://bucket/path/transcript.json",
            "recording_path": "",
            "api_keys": {},
        }
        response = await client.post(url, json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "recording_path cannot be empty" in response.json()["detail"]
        
        payload = {
            "voice_eval_result": "s3://bucket/path/result.json",
            "transcript_path": "",
            "recording_path": "s3://bucket/path/recording.ogg",
            "api_keys": {},
        }
        response = await client.post(url, json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "transcript_path cannot be empty" in response.json()["detail"]
        
        payload = {
            "voice_eval_result": "",
            "transcript_path": "s3://bucket/path/transcript.json",
            "recording_path": "s3://bucket/path/recording.ogg",
            "api_keys": {},
        }
        response = await client.post(url, json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "voice_eval_result cannot be empty" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_post_voice_evaluation_with_missing_bucket(
        self, client, test_ids
    ):
        url = f"/internal/api/v1/agents/{test_ids['tenant_id']}/{test_ids['agent_id']}/{test_ids['run_id']}/evaluation/voice"
        
        payload = {
            "voice_eval_result": "just_a_path",
            "transcript_path": "s3://bucket/path/transcript.json",
            "recording_path": "s3://bucket/path/recording.ogg",
            "api_keys": {},
        }
        response = await client.post(url, json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "must include bucket name and path" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_post_voice_evaluation_conflict_without_force(
        self, client, test_ids, valid_voice_eval_payload
    ):
        url = f"/internal/api/v1/agents/{test_ids['tenant_id']}/{test_ids['agent_id']}/{test_ids['run_id']}/evaluation/voice"
        
        payload_with_force = {**valid_voice_eval_payload, "force": True}
        response1 = await client.post(url, json=payload_with_force)
        assert response1.status_code == status.HTTP_200_OK
        
        response2 = await client.post(url, json=valid_voice_eval_payload)
        assert response2.status_code == status.HTTP_409_CONFLICT
        assert "already requested" in response2.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_post_voice_evaluation_force_recreate(
        self, client, test_ids, valid_voice_eval_payload
    ):
        url = f"/internal/api/v1/agents/{test_ids['tenant_id']}/{test_ids['agent_id']}/{test_ids['run_id']}/evaluation/voice"
        
        payload_with_force = {**valid_voice_eval_payload, "force": True}
        
        response = await client.post(url, json=payload_with_force)
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_post_voice_evaluation_with_custom_model(
        self, client, test_ids, valid_voice_eval_payload
    ):
        url = f"/internal/api/v1/agents/{test_ids['tenant_id']}/{test_ids['agent_id']}/{test_ids['run_id']}/evaluation/voice"
        
        await client.delete(url)
        
        payload = {
            **valid_voice_eval_payload,
            "evaluation_model": "openai/gpt-4o-mini",
            "force": True,
        }
        
        response = await client.post(url, json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] in ["Requested", "InProgress", "Completed", "Failed"]

    @pytest.mark.asyncio
    async def test_get_voice_evaluation_not_found(self, client):
        url = f"/internal/api/v1/agents/nonexistent/nonexistent/nonexistent/evaluation/voice"
        
        response = await client.get(url)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_voice_evaluation_success(
        self, client, test_ids, valid_voice_eval_payload
    ):
        url = f"/internal/api/v1/agents/{test_ids['tenant_id']}/{test_ids['agent_id']}/{test_ids['run_id']}/evaluation/voice"
        
        await client.delete(url)
        
        payload_with_force = {**valid_voice_eval_payload, "force": True}
        post_response = await client.post(url, json=payload_with_force)
        assert post_response.status_code == status.HTTP_200_OK
        
        get_response = await client.get(url)
        assert get_response.status_code == status.HTTP_200_OK
        
        data = get_response.json()
        assert "status" in data
        assert "progress" in data
        assert data["status"] in ["Requested", "InProgress", "Completed", "Failed"]

    @pytest.mark.asyncio
    async def test_delete_voice_evaluation_not_found(self, client):
        url = f"/internal/api/v1/agents/nonexistent/nonexistent/nonexistent/evaluation/voice"
        
        response = await client.delete(url)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_voice_evaluation_success(
        self, client, test_ids, valid_voice_eval_payload
    ):
        url = f"/internal/api/v1/agents/{test_ids['tenant_id']}/{test_ids['agent_id']}/{test_ids['run_id']}/evaluation/voice"
        
        await client.delete(url)
        
        payload_with_force = {**valid_voice_eval_payload, "force": True}
        post_response = await client.post(url, json=payload_with_force)
        assert post_response.status_code == status.HTTP_200_OK
        
        delete_response = await client.delete(url)
        assert delete_response.status_code == status.HTTP_200_OK
        
        get_response = await client.get(url)
        assert get_response.status_code == status.HTTP_404_NOT_FOUND
