from typing import Optional
import json

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status
from fastapi.responses import JSONResponse

from deusauditron.app_logging.context import set_logging_context
from deusauditron.app_logging.logger import logger
from deusauditron.state.manager import StateManager
from deusauditron.util.helper import Trinity

from deusauditron.schemas.autogen.references.voice_evaluation_result_schema import (
    VoiceEvaluationResult,
    VoiceEvalStatus,
)
from deusauditron.schemas.shared_models.models import VoiceEvalRequest
from deusauditron.schemas.shared_models.models import VoiceEvaluationPayload

from deusauditron.engine import Auditron
from deusauditron.config import TracingManager
from deusauditron.state.object.eval import VoiceEvalState


voice_evaluation_router = APIRouter(
    prefix="/agents/{tenant_id}/{agent_id}/{run_id}/evaluation/voice",
    tags=["voice_evaluation"],
    responses={404: {"description": "Not found"}},
)


@voice_evaluation_router.post("")
async def evaluate_agent_voice(
    tenant_id: str, agent_id: str, run_id: str, payload: VoiceEvaluationPayload
):
    """Voice evaluation endpoint for STT and TTS evaluation."""
    if TracingManager().is_enabled:
        with TracingManager().get_tracer().start_as_current_span(f"voice_eval/{tenant_id}/{agent_id}/{run_id}", # type: ignore
                                                                 openinference_span_kind="agent"): # type: ignore
            return await evaluate_agent_voice_internal(tenant_id, agent_id, run_id, payload)
    
    return await evaluate_agent_voice_internal(tenant_id, agent_id, run_id, payload)
    

def _validate_s3_path(path: str, field_name: str) -> None:
    """Validate S3 path format"""
    if not path or not path.strip():
        raise ValueError(f"{field_name} cannot be empty")
    
    # Check if path contains bucket name
    cleaned_path = path.replace("s3://", "")
    parts = cleaned_path.split("/")
    
    if len(parts) < 2:
        raise ValueError(f"{field_name} must include bucket name and path (e.g., 'bucket/path/to/file' or 's3://bucket/path/to/file')")
    
    bucket_name = parts[0]
    if not bucket_name or not bucket_name.strip():
        raise ValueError(f"{field_name} must have a valid bucket name")


async def evaluate_agent_voice_internal(tenant_id: str, agent_id: str, run_id: str, payload: VoiceEvaluationPayload):
    set_logging_context(tenant_id, agent_id, run_id)

    try:
        engine = Auditron.get_instance()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        # Validate S3 paths
        _validate_s3_path(payload.recording_path, "recording_path")
        _validate_s3_path(payload.transcript_path, "transcript_path")
        _validate_s3_path(payload.voice_eval_result, "voice_eval_result")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    try:
        voice_eval_state = await StateManager().get_voice_eval_state(tenant_id, agent_id, run_id)
        if voice_eval_state is not None:
            voice_eval_result = voice_eval_state.get_voice_evaluation_result()
            result = await _handle_existing_voice_evaluation(
                voice_eval_result, payload.force, tenant_id, agent_id, run_id, engine
            )
            if result is not None:
                return result

        voice_eval_result = VoiceEvaluationResult(
            status=VoiceEvalStatus.Requested,
            progress=0,
            start_time=None,
            end_time=None,
            stt_evaluation=None,
            tts_evaluation=None,
            overall_accuracy=None,
            error_message=None,
            metadata={}
        )

        voice_eval_state = VoiceEvalState(voice_evaluation_result=voice_eval_result)
        
        request = VoiceEvalRequest(
            tenant_id=tenant_id, agent_id=agent_id, run_id=run_id, payload=payload
        )
        
        await engine.submit_voice_eval_request(request, voice_eval_state)
        voice_eval_result = voice_eval_state.get_voice_evaluation_result()
        return voice_eval_result.model_dump() if voice_eval_result else {}
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"## Unexpected error processing voice evaluation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def _handle_existing_voice_evaluation(voice_eval_result, force_flag, tenant_id, agent_id, run_id, engine):
    force = force_flag is True
    if not force or (voice_eval_result and voice_eval_result.status not in (VoiceEvalStatus.Completed, VoiceEvalStatus.Error)):
        logger.warning(
            f"Voice evaluation conflict for {tenant_id}/{agent_id}/{run_id} - force: {force}, status: {voice_eval_result.status if voice_eval_result else 'None'}"
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": f"Voice evaluation already requested for {tenant_id}/{agent_id}/{run_id}. If completed/error and to re-evaluate, pass force=true",
                "current_status": voice_eval_result.status.value if voice_eval_result and voice_eval_result.status else "Unknown",
            },
        )

    logger.info(
        f"Force creating new voice evaluation for {tenant_id}/{agent_id}/{run_id}, deleting existing"
    )
    await engine.delete_voice_eval_request(
        VoiceEvalRequest(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id)
    )


@voice_evaluation_router.get("")
async def get_voice_evaluation(
    tenant_id: str,
    agent_id: str,
    run_id: str,
    voice_eval_path: Optional[str] = None,
) -> Optional[dict]:
    set_logging_context(tenant_id, agent_id, run_id)

    try:
        eval_state = await StateManager().get_voice_eval_state(tenant_id, agent_id, run_id)
        if eval_state is None:
            if voice_eval_path:
                try:
                    logger.info(f"Attempting to read voice evaluation result from S3: {voice_eval_path}")
                    file_content = await Trinity.aread_from_s3(voice_eval_path)
                    if file_content:
                        eval_dict = json.loads(file_content)
                        return VoiceEvaluationResult(**eval_dict).model_dump()
                except Exception as e:
                    logger.error(f"Error reading voice evaluation from S3: {e}")
            
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice evaluation not found for {tenant_id}/{agent_id}/{run_id}",
            )

        voice_eval_result = eval_state.get_voice_evaluation_result()
        return voice_eval_result.model_dump() if voice_eval_result else {}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"## Unexpected error retrieving voice evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@voice_evaluation_router.delete("")
async def delete_voice_evaluation(tenant_id: str, agent_id: str, run_id: str):
    set_logging_context(tenant_id, agent_id, run_id)

    try:
        engine = Auditron.get_instance()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        voice_eval_state = await StateManager().get_voice_eval_state(tenant_id, agent_id, run_id)
        if voice_eval_state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice evaluation not found for {tenant_id}/{agent_id}/{run_id}",
            )

        voice_eval_result = voice_eval_state.get_voice_evaluation_result()
        if voice_eval_result and voice_eval_result.status == VoiceEvalStatus.In_Progress:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot delete voice evaluation for {tenant_id}/{agent_id}/{run_id} while it's in progress",
            )

        await engine.delete_voice_eval_request(
            VoiceEvalRequest(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"## Unexpected error processing voice evaluation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
