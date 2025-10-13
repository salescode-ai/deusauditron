import os
import uuid
import json
from phoenix.client import Client
import nest_asyncio
from typing import Dict, Any, List, Optional
import httpx

from fastapi import APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from deusauditron.config import get_config


scenario_evaluation_router = APIRouter(
    prefix="/scenario/run",
    tags=["scenario_evaluation"],
    responses={404: {"description": "Not found"}},
)

client = Client(
    base_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"), 
    api_key=os.getenv("PHOENIX_API_KEY")
)


@scenario_evaluation_router.post("")
async def run_scenario(
    payload: ScenarioPayload,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
):
    try:
        nest_asyncio.apply()
        dm_adapter = DMAdapter()

        metadata = payload.metadata
        blueprint = payload.blueprint
        if not blueprint:
            if credentials is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication credentials are required"
                )
            auth_header = f"Bearer {credentials.credentials.removeprefix('Bearer ')}"
            agent_name = payload.agent_name.split("/")[-2]
            response = httpx.get(
                f"{get_config().mgmt_url}/agents/{agent_name}", 
                headers={"Authorization": auth_header}
            )
            response.raise_for_status()
            agent_config = response.json()
            blueprint = agent_config.get("blueprint", "")
        if not blueprint:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Blueprint not found for agent: {payload.agent_name}"
            )
        
        await validate_agent(metadata, blueprint)

        async def scenario_task(example: Example, fallback_last_node_name: str) -> str:
            try:
                transcript_dicts = json.loads(str(example.input.get("Input", ""))).get("messages", [])
                transcript = [Message(**msg_dict) for msg_dict in transcript_dicts]
                dataset_metadata = json.loads(str(example.metadata.get("Meta Data", "")))
                catalogs = dataset_metadata.get("catalogs", {})
                last_node_name = dataset_metadata.get("last_node_name", "")
                dynamic_data: Dict[str, str] = {}
                for key, value in catalogs.items():
                    dynamic_data[key] = json.dumps(value, indent=2, ensure_ascii=False)

                final_output = await run_task(
                    metadata=metadata,
                    blueprint=blueprint,
                    dynamic_data=dynamic_data,
                    replay=payload.replay,
                    transcript=transcript,
                    last_node_name=last_node_name,
                    fallback_last_node_name=fallback_last_node_name,
                )
                return final_output
            except Exception as e:
                logger.error(f"Error in scenario_task: {str(e)}")
                return f"ERROR: {str(e)}"

        async def scenario_evaluator(output, expected) -> EvaluationResult:
            try:
                expected_output = expected.get("Output", "")
                final_output = output

                if isinstance(final_output, str) and final_output.startswith("ERROR:"):
                    error_message = final_output.replace("ERROR:", "").strip()
                    return EvaluationResult(
                        score=0.0,
                        label="ERROR",
                        explanation=f"Task execution failed: {error_message}",
                    )

                response = await EvaluationUtils().custom_evaluator(final_output, expected_output, dm_adapter)
                evaluation_result = EvaluationResult(
                    score=1.0 if response.get("result", "UNDEFINED") == "PASS" else 0.0,
                    label=response.get("result", "UNDEFINED"),
                    explanation=response.get("reasoning", "UNDEFINED"),
                )
                return evaluation_result
            except Exception as e:
                logger.error(f"Error in scenario_evaluator: {str(e)}")
                return EvaluationResult(
                    score=0.0,
                    label="ERROR",
                    explanation=f"Evaluation failed: {str(e)}",
                )
    
        if len(payload.dataset_names) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset names are required"
            )

        experiment_ids = []
        for dataset_name in payload.dataset_names:
            dataset = client.datasets.get_dataset(dataset=dataset_name)

            async def scenario_task_helper(example: Example) -> str:
                return await scenario_task(example, dataset.metadata.get("last_node_name", ""))

            experiment = run_experiment(
                dataset=dataset,
                task=scenario_task_helper,
                evaluators=[scenario_evaluator],
                experiment_metadata={
                    "agent_name": payload.agent_name,
                    "experiment_name": payload.experiment_name
                },
            )
            experiment_ids.append(experiment.id)
        return {"success": True, "experiment_ids": experiment_ids}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {e}"
        )
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phoenix service is unavailable. Please ensure the service is running and You have set the PHOENIX_COLLECTOR_ENDPOINT and PHOENIX_API_KEY environment variables."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_scenario: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during scenario evaluation"
        )


async def run_task(
    metadata: Dict[str, Any],
    blueprint: str,
    dynamic_data: Dict[str, str],
    replay: bool,
    transcript: List[Message],
    last_node_name: str,
    fallback_last_node_name: str,
):
    tenant_id = str(uuid.uuid4())
    agent_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    set_logging_context(tenant_id, agent_id, run_id)

    dm_adapter = None
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
            if not last_node_name:
                last_node_name = messages[-1].metadata.get("node", "") if messages else ""
            if not last_node_name:
                last_node_name = fallback_last_node_name
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

        if current_output is None:
            raise Exception("No output generated from agent execution")

        final_output = current_output[0]["content"]
        return final_output
    except Exception as e:
        logger.error(f"Error in run_task: {str(e)}")
        raise e
    finally:
        if dm_adapter:
            try:
                await dm_adapter.delete_agent(
                    tenant_id=tenant_id, agent_id=agent_id, run_id=run_id
                )
            except Exception as e:
                logger.info(f"Falied to delete agent: {str(e)}")

async def validate_agent(metadata: Dict[str, Any], blueprint: str):
    try:
        response = httpx.post(
            get_config().deusmachina_url + "/schema/validate",
            json={
                "metadata": metadata,
                "blueprint": blueprint,
                "api_keys": {
                    "groq": os.getenv("GROQ_API_KEY"),
                    "openai": os.getenv("OPENAI_API_KEY"),
                }
            }
        )
        response.raise_for_status()
        validation_result = response.json()
        if validation_result.get("overall_passed", False) == True:
            return True
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=validation_result
            )
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Deusmachina service is unavailable. Please ensure the service is running."
        )
    except Exception as e:
        logger.error(f"Unexpected error during agent validation: {str(e)}")
        raise e