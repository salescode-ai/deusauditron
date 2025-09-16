import uuid

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from app_logging.context import set_logging_context
from schemas.shared_models.models import ScenarioPayload
from deusmachine_adapter.dm_adapter import DMAdapter
from eval.eval_utils import EvaluationUtils
from schemas.shared_models.models import MessageRole
from app_logging.logger import logger


scenario_evaluation_router = APIRouter(
    prefix="/scenario/run",
    tags=["scenario_evaluation"],
    responses={404: {"description": "Not found"}},
)


@scenario_evaluation_router.post("")
async def run_scenario(payload: ScenarioPayload):
    tenant_id = str(uuid.uuid4())
    agent_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    set_logging_context(tenant_id, agent_id, run_id)

    try:
        dm_adapter = DMAdapter()
        await dm_adapter.create_agent(
            tenant_id=tenant_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=payload.metadata,
            blueprint=payload.blueprint,
            dynamic_data=payload.dynamic_data,
        )
        logger.info(f"Agent created: {tenant_id}/{agent_id}/{run_id}")

        replay = payload.replay
        current_output = None
        
        if not replay:
            transcript = payload.transcript
            last_user_input = (
                transcript[-1].content
                if isinstance(transcript[-1].content, str)
                else ""
            )
            messages = transcript[:-1]
            last_node_name = messages[-1].metadata.get("node", "")
            current_output = await dm_adapter.run_agent(
                tenant_id=tenant_id,
                agent_id=agent_id,
                run_id=run_id,
                user_input=last_user_input,
                messages=messages,
                entry_node_name=last_node_name,
            )
        else:
            user_messages = [m for m in payload.transcript if m.role == MessageRole.USER]
            for message in user_messages:
                current_output = await dm_adapter.run_agent(
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    user_input=message.content if isinstance(message.content, str) else "",
                )

        await dm_adapter.delete_agent(
            tenant_id=tenant_id, agent_id=agent_id, run_id=run_id
        )

        if current_output is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No output generated from agent execution"
            )

        final_output = current_output[0]["content"]
        response = await EvaluationUtils().custom_evaluator(final_output, payload.expected_output, dm_adapter)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
