import os
import uuid
import json
import phoenix as px
import nest_asyncio
from typing import Dict, Any, List

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status
from phoenix.experiments import run_experiment
from phoenix.experiments.types import Example
from phoenix.experiments.types import EvaluationResult

from deusauditron.app_logging.context import set_logging_context
from deusauditron.schemas.shared_models.models import ScenarioPayload, Message
from deusauditron.deusmachine_adapter.dm_adapter import DMAdapter
from deusauditron.eval.eval_utils import EvaluationUtils
from deusauditron.schemas.shared_models.models import MessageRole
from deusauditron.app_logging.logger import logger


scenario_evaluation_router = APIRouter(
    prefix="/scenario/run",
    tags=["scenario_evaluation"],
    responses={404: {"description": "Not found"}},
)

client = px.Client(
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"), 
    api_key=os.getenv("PHOENIX_API_KEY")
)


@scenario_evaluation_router.post("")
async def run_scenario(payload: ScenarioPayload):
    nest_asyncio.apply()
    
    dataset = client.get_dataset(name=payload.dataset_name)
    dm_adapter = DMAdapter()

    async def scenario_task(example: Example) -> str:
        transcript_dicts = json.loads(str(example.input.get("Input", ""))).get("messages", [])
        transcript = [Message(**msg_dict) for msg_dict in transcript_dicts]
        metadata = json.loads(str(example.metadata.get("Meta Data", ""))).get("catalogs", {})
        dynamic_data: Dict[str, str] = {}
        for key, value in metadata.items():
            dynamic_data[key] = json.dumps(value, indent=2, ensure_ascii=False)

        final_output = await run_task(
            metadata=payload.metadata,
            blueprint=payload.blueprint,
            dynamic_data=dynamic_data,
            replay=payload.replay,
            transcript=transcript,
        )
        return final_output

    async def scenario_evaluator(output, expected) -> EvaluationResult:
        expected_output = expected.get("Output", "")
        final_output = output

        response = await EvaluationUtils().custom_evaluator(final_output, expected_output, dm_adapter)
        evaluation_result = EvaluationResult(
            score=1.0 if response.get("result", "UNDEFINED") == "PASS" else 0.0,
            label=response.get("result", "UNDEFINED"),
            explanation=response.get("reasoning", "UNDEFINED"),
        )
        return evaluation_result
    
    try:
        experiment = run_experiment(
            dataset=dataset,
            task=scenario_task,
            evaluators=[scenario_evaluator],
            experiment_metadata={
                "agent_name": payload.agent_name,
                "experiment_name": payload.experiment_name
            },
        )
        return {"success": True, "experiment_id": experiment.id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def run_task(
    metadata: Dict[str, Any],
    blueprint: str,
    dynamic_data: Dict[str, str],
    replay: bool,
    transcript: List[Message],
):
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
            metadata=metadata,
            blueprint=blueprint,
            dynamic_data=dynamic_data,
        )
        logger.info(f"Agent created: {tenant_id}/{agent_id}/{run_id}")

        replay = replay
        current_output = None
        
        if not replay:
            last_user_input = (
                transcript[-1].content
                if isinstance(transcript[-1].content, str)
                else ""
            )
            messages = transcript[:-1]
            last_node_name = messages[-1].metadata.get("node", "") if messages else ""
            current_output = await dm_adapter.run_agent(
                tenant_id=tenant_id,
                agent_id=agent_id,
                run_id=run_id,
                user_input=last_user_input,
                messages=messages,
                entry_node_name=last_node_name,
            )
        else:
            user_messages = [m for m in transcript if m.role == MessageRole.USER]
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
        return final_output
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
