import os
import httpx
from fastapi import Query

class PhoenixClient:
    def __init__(self):
        self.base_url = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        self.api_key = os.getenv("PHOENIX_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def get_http_client(self):
        return httpx.AsyncClient(headers=self.headers)
    
    def get_experiments_url(self, dataset_id: str) -> str:
        return f"{self.base_url}/v1/datasets/{dataset_id}/experiments"
    
    def get_experiment_url(self, experiment_id: str) -> str:
        return f"{self.base_url}/v1/experiments/{experiment_id}/json"

def get_datasets_helper(client, filter: str = Query(default="")):
    all_datasets = client.datasets.list()
    filtered_datasets = []
    for dataset in all_datasets:
        if dataset["name"].startswith(filter) or dataset["name"].startswith("common/"):
            filtered_datasets.append(dataset)
    return filtered_datasets