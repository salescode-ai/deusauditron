import os
import httpx
from phoenix.client import Client

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

from deusauditron.app_logging.logger import logger
from deusauditron.schemas.shared_models.models import AppendDatasetRowsPayload

phoenix_router = APIRouter(
    prefix="/phoenix",
    tags=["phoenix"],
    responses={404: {"description": "Not found"}},
)

client = Client(
    base_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
    api_key=os.getenv("PHOENIX_API_KEY"),
)

@phoenix_router.get("/datasets")
async def get_datasets(filter: str = Query(default="")):
    try:
        all_datasets = client.datasets.list()
        filtered_datasets = []
        for dataset in all_datasets:
            if dataset["name"].startswith(filter):
                filtered_datasets.append(dataset)
        return {"success": True, "datasets": filtered_datasets}
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@phoenix_router.get("/datasets/{dataset_id}/rows")
async def get_dataset_rows(dataset_id: str):
    try:
        dataset = client.datasets.get_dataset(dataset=dataset_id)
        return {"success": True, "rows": dataset.examples}
    except Exception as e:
        logger.error(f"Error getting dataset rows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@phoenix_router.post("/datasets/{dataset_id}/rows/append")
async def append_dataset_rows(dataset_id: str, payload: AppendDatasetRowsPayload):
    try:
        inputs = [{"Input": input} for input in payload.inputs]
        outputs = [{"Output": output} for output in payload.outputs]
        metadata_list = [{"Meta Data": metadata} for metadata in payload.metadata]
        client.datasets.add_examples_to_dataset(
            dataset=dataset_id,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata_list
        )
        return {"success": True, "message": "Dataset rows appended successfully"}
    except Exception as e:
        logger.error(f"Error appending dataset rows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@phoenix_router.get("/experiments")
async def get_experiments(filter: str = Query(default="")):
    try:
        all_datasets = client.datasets.list()
        filtered_datasets = []
        for dataset in all_datasets:
            if dataset["name"].startswith(filter):
                filtered_datasets.append(dataset)
        
        all_experiments = []
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('PHOENIX_API_KEY')}"
        }
        async with httpx.AsyncClient(headers=headers) as client_http:
            for dataset in filtered_datasets:
                base_url = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
                experiments_url = f"{base_url}/v1/datasets/{dataset['id']}/experiments"
                response = await client_http.get(experiments_url)
                if response.status_code == 200:
                    experiments_data = response.json()["data"]
                    for experiment in experiments_data:
                        if experiment.get("metadata", {}).get("agent_name", "") == filter:
                            all_experiments.append(experiment)
                else:
                    logger.warning(f"Failed to get experiments for dataset {dataset['id']}: {response.status_code}")
        
        return {"success": True, "experiments": all_experiments}
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))