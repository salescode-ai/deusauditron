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

from deusauditron.schemas.autogen.references.evaluation_result_schema import (
    EvaluationResult,
    Status,
)
from deusauditron.schemas.shared_models.models import AgentEvalRequest
from deusauditron.schemas.shared_models.models import EvaluationPayload
from deusauditron.schemas.autogen.references.granular_evaluation_result_schema import (
    GranularEvaluationResults,
)
from deusauditron.state.object.eval import EvalState

from deusauditron.engine import Auditron

from deusauditron.config import TracingManager


evaluation_router = APIRouter(
    prefix="/agents/{tenant_id}/{agent_id}/{run_id}/evaluation",
    tags=["evaluation"],
    responses={404: {"description": "Not found"}},
)


@evaluation_router.post("")
async def evaluate_agent(
    tenant_id: str, agent_id: str, run_id: str, payload: EvaluationPayload
):
    if TracingManager().is_enabled:
        with TracingManager().get_tracer().start_as_current_span(f"run/{tenant_id}/{agent_id}/{run_id}", # type: ignore
                                                                 openinference_span_kind="agent"): # type: ignore
            return await evaluate_agent_internal(tenant_id, agent_id, run_id, payload)
    else:
        return await evaluate_agent_internal(tenant_id, agent_id, run_id, payload)
    
async def evaluate_agent_internal(tenant_id: str, agent_id: str, run_id: str, payload: EvaluationPayload):
    set_logging_context(tenant_id, agent_id, run_id)

    try:
        engine = Auditron.get_instance()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        state = await StateManager().get_state(tenant_id, agent_id, run_id)
        if state is None:
            logger.warning("Agent state not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found for {tenant_id}/{agent_id}/{run_id}",
            )

        eval_state = await StateManager().get_eval_state(tenant_id, agent_id, run_id)
        if eval_state is not None:
            eval_result = eval_state.get_evaluation_result()
            result = await _handle_existing_evaluation(
                eval_result, payload.force, tenant_id, agent_id, run_id, engine
            )
            if result is not None:
                return result

        eval_result = EvaluationResult(
            status=Status.Requested,
            progress=0,
            turn_level_evaluations=[],
            node_level_evaluations=[],
            flow_level_evaluations=[],
            intent_level_evaluations=[],
            auto_refinements=[],
            start_time=None,
            end_time=None,
            evaluated_nodes=[],
        )
        eval_state = EvalState(evaluation_result=eval_result)
        request = AgentEvalRequest(
            tenant_id=tenant_id, agent_id=agent_id, run_id=run_id, payload=payload
        )
        await engine.submit_eval_request(request, eval_state)
        return eval_state.get_evaluation_result()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"## Unexpected error processing evaluation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def _handle_existing_evaluation(eval_result, force_flag, tenant_id, agent_id, run_id, engine):
    force = force_flag is True
    if not force or (eval_result and eval_result.status not in (Status.Completed, Status.Error)):
        logger.warning(
            f"Evaluation conflict for {tenant_id}/{agent_id}/{run_id} - force: {force}, status: {eval_result.status if eval_result else 'None'}"
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": f"Evaluation already requested for {tenant_id}/{agent_id}/{run_id}. If completed/error and to re-evaluate, pass force=true",
                "current_status": eval_result.status.value if eval_result and eval_result.status else "Unknown",
            },
        )

    logger.info(
        f"Force creating new evaluation for {tenant_id}/{agent_id}/{run_id}, deleting existing"
    )
    await engine.delete_eval_request(
        AgentEvalRequest(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id)
    )


@evaluation_router.get("")
async def get_evaluation(
    tenant_id: str,
    agent_id: str,
    run_id: str,
    evaluation_path: Optional[str] = None,
) -> Optional[EvaluationResult]:
    set_logging_context(tenant_id, agent_id, run_id)

    try:
        engine = Auditron.get_instance()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        state = await StateManager().get_state(tenant_id, agent_id, run_id)
        if state is None:
            logger.warning("Agent state not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found for {tenant_id}/{agent_id}/{run_id}",
            )

        eval_state = await StateManager().get_eval_state(tenant_id, agent_id, run_id)
        if eval_state is not None and eval_state.get_evaluation_result() is not None:
            eval_result = eval_state.get_evaluation_result()
            logger.info(
                f"Found evaluation result in Redis for {tenant_id}/{agent_id}/{run_id}"
            )
            return eval_result

        if evaluation_path:
            logger.info(
                f"Evaluation not found in Redis, trying S3 for {tenant_id}/{agent_id}/{run_id}"
            )
            try:
                s3_url = f"s3://{evaluation_path}"
                s3_content = Trinity.read_from_s3(s3_url)
                if s3_content:
                    eval_data = json.loads(s3_content)
                    eval_result = EvaluationResult(**eval_data)
                    logger.info(
                        f"Successfully retrieved evaluation from S3 for {tenant_id}/{agent_id}/{run_id}"
                    )
                    return eval_result
            except Exception as e:
                logger.warning(
                    f"Error reading from S3 for {tenant_id}/{agent_id}/{run_id}: {e}"
                )

        logger.warning(
            f"Evaluation was never requested for {tenant_id}/{agent_id}/{run_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation never requested for {tenant_id}/{agent_id}/{run_id}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"## Unexpected error processing evaluation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@evaluation_router.delete("")
async def delete_evaluation(tenant_id: str, agent_id: str, run_id: str):
    set_logging_context(tenant_id, agent_id, run_id)
    logger.info("## Received evaluation deletion request (Deusauditron)")

    try:
        engine = Auditron.get_instance()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        state = await StateManager().get_state(tenant_id, agent_id, run_id)
        if state is None:
            logger.warning("Agent state not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found for {tenant_id}/{agent_id}/{run_id}",
            )
        eval_state = await StateManager().get_eval_state(tenant_id, agent_id, run_id)
        if eval_state is None or eval_state.get_evaluation_result() is None:
            logger.warning(
                f"Evaluation was never requested for {tenant_id}/{agent_id}/{run_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation never requested for {tenant_id}/{agent_id}/{run_id}",
            )
        evaluation_result = eval_state.get_evaluation_result()
        if evaluation_result and evaluation_result.status not in (
            Status.Completed,
            Status.Error,
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Evaluation is progress, please wait for completion, then retry deletion",
            )
        await engine.delete_eval_request(
            AgentEvalRequest(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"## Unexpected error processing evaluation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

