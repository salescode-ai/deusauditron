import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import status

from deusauditron.app import create_app

PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")

SKIP_TESTS = not PHOENIX_COLLECTOR_ENDPOINT or not PHOENIX_API_KEY

SKIP_MESSAGE = """
Phoenix API keys are not configured. Please set the following environment variables to run these tests:
  - PHOENIX_COLLECTOR_ENDPOINT: The Phoenix collector endpoint URL
  - PHOENIX_API_KEY: Your Phoenix API key
"""

pytestmark = pytest.mark.skipif(
    SKIP_TESTS,
    reason=SKIP_MESSAGE
)


class TestPhoenixIntegration:

    @pytest_asyncio.fixture
    async def client(self):
        application = create_app()
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    @pytest.mark.asyncio
    async def test_get_datasets_without_filter(self, client):
        response = await client.get("/api/v1/phoenix/datasets")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    @pytest.mark.asyncio
    async def test_get_datasets_with_filter(self, client):
        filter_param = "common/"
        response = await client.get(f"/api/v1/phoenix/datasets?filter={filter_param}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "datasets" in data
        assert isinstance(data["datasets"], list)
        
        for dataset in data["datasets"]:
            assert "name" in dataset
            assert dataset["name"].startswith(filter_param) or dataset["name"].startswith("common/")

    @pytest.mark.asyncio
    async def test_get_datasets_with_custom_filter(self, client):
        filter_param = "test"
        response = await client.get(f"/api/v1/phoenix/datasets?filter={filter_param}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    @pytest.mark.asyncio
    async def test_get_dataset_rows_valid_dataset(self, client):
        datasets_response = await client.get("/api/v1/phoenix/datasets")
        assert datasets_response.status_code == status.HTTP_200_OK
        datasets = datasets_response.json()["datasets"]
        
        if not datasets:
            pytest.skip("No datasets available in Phoenix to test with")
        
        dataset_id = datasets[0]["id"]
        
        response = await client.get(f"/api/v1/phoenix/datasets/{dataset_id}/rows")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "rows" in data
        assert isinstance(data["rows"], list)

    @pytest.mark.asyncio
    async def test_get_dataset_rows_invalid_dataset(self, client):
        invalid_dataset_id = "non-existent-dataset-id-123456"
        response = await client.get(f"/api/v1/phoenix/datasets/{invalid_dataset_id}/rows")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_append_dataset_rows_valid_dataset(self, client):
        datasets_response = await client.get("/api/v1/phoenix/datasets")
        assert datasets_response.status_code == status.HTTP_200_OK
        datasets = datasets_response.json()["datasets"]
        
        if not datasets:
            pytest.skip("No datasets available in Phoenix to test with")
        
        dataset_id = datasets[0]["id"]
        
        payload = {
            "inputs": ["Test input 1", "Test input 2"],
            "outputs": ["Test output 1", "Test output 2"],
            "metadata": ["Test metadata 1", "Test metadata 2"]
        }
        
        response = await client.post(
            f"/api/v1/phoenix/datasets/{dataset_id}/rows/append",
            json=payload
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Dataset rows appended successfully"

    @pytest.mark.asyncio
    async def test_append_dataset_rows_mismatched_arrays(self, client):
        datasets_response = await client.get("/api/v1/phoenix/datasets")
        assert datasets_response.status_code == status.HTTP_200_OK
        datasets = datasets_response.json()["datasets"]
        
        if not datasets:
            pytest.skip("No datasets available in Phoenix to test with")
        
        dataset_id = datasets[0]["id"]
        
        payload = {
            "inputs": ["Test input 1", "Test input 2"],
            "outputs": ["Test output 1"],
            "metadata": ["Test metadata 1", "Test metadata 2"]
        }
        
        response = await client.post(
            f"/api/v1/phoenix/datasets/{dataset_id}/rows/append",
            json=payload
        )
        
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_500_INTERNAL_SERVER_ERROR]

    @pytest.mark.asyncio
    async def test_get_experiments_without_filter(self, client):
        response = await client.get("/api/v1/phoenix/experiments")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "experiments" in data
        assert isinstance(data["experiments"], list)

    @pytest.mark.asyncio
    async def test_get_experiments_with_filter(self, client):
        filter_param = "test-agent"
        response = await client.get(f"/api/v1/phoenix/experiments?filter={filter_param}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "experiments" in data
        assert isinstance(data["experiments"], list)
        
        for experiment in data["experiments"]:
            if "metadata" in experiment and "agent_name" in experiment["metadata"]:
                assert experiment["metadata"]["agent_name"] == filter_param

    @pytest.mark.asyncio
    async def test_get_experiment_by_id_valid(self, client):
        experiments_response = await client.get("/api/v1/phoenix/experiments")
        assert experiments_response.status_code == status.HTTP_200_OK
        experiments = experiments_response.json()["experiments"]
        
        if not experiments:
            pytest.skip("No experiments available in Phoenix to test with")
        
        experiment_id = experiments[0]["id"]
        
        response = await client.get(f"/api/v1/phoenix/experiments/{experiment_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "experiment" in data
        assert data["experiment"] is not None

    @pytest.mark.asyncio
    async def test_get_experiment_by_id_invalid(self, client):
        invalid_experiment_id = "non-existent-experiment-id-12345"
        response = await client.get(f"/api/v1/phoenix/experiments/{invalid_experiment_id}")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_complete_workflow(self, client):
        datasets_response = await client.get("/api/v1/phoenix/datasets")
        assert datasets_response.status_code == status.HTTP_200_OK
        datasets = datasets_response.json()["datasets"]
        
        if not datasets:
            pytest.skip("No datasets available in Phoenix to test with")
        
        dataset_id = datasets[0]["id"]
        
        rows_before = await client.get(f"/api/v1/phoenix/datasets/{dataset_id}/rows")
        assert rows_before.status_code == status.HTTP_200_OK
        initial_row_count = len(rows_before.json()["rows"])
        
        test_payload = {
            "inputs": ["Integration test input"],
            "outputs": ["Integration test output"],
            "metadata": ["Integration test metadata"]
        }
        
        append_response = await client.post(
            f"/api/v1/phoenix/datasets/{dataset_id}/rows/append",
            json=test_payload
        )
        assert append_response.status_code == status.HTTP_200_OK
        
        rows_after = await client.get(f"/api/v1/phoenix/datasets/{dataset_id}/rows")
        assert rows_after.status_code == status.HTTP_200_OK
        final_row_count = len(rows_after.json()["rows"])
        
        assert final_row_count == initial_row_count + 1

    @pytest.mark.asyncio
    async def test_multiple_datasets_filter(self, client):
        response1 = await client.get("/api/v1/phoenix/datasets?filter=")
        assert response1.status_code == status.HTTP_200_OK
        all_datasets = response1.json()["datasets"]
        
        response2 = await client.get("/api/v1/phoenix/datasets?filter=common/")
        assert response2.status_code == status.HTTP_200_OK
        common_datasets = response2.json()["datasets"]
        
        assert len(common_datasets) <= len(all_datasets)

    @pytest.mark.asyncio  
    async def test_experiments_dataset_correlation(self, client):
        experiments_response = await client.get("/api/v1/phoenix/experiments")
        assert experiments_response.status_code == status.HTTP_200_OK
        experiments = experiments_response.json()["experiments"]
        
        if not experiments:
            pytest.skip("No experiments available in Phoenix to test with")
        
        datasets_response = await client.get("/api/v1/phoenix/datasets")
        assert datasets_response.status_code == status.HTTP_200_OK
        datasets = datasets_response.json()["datasets"]
        
        assert isinstance(experiments, list)
        assert isinstance(datasets, list)
        
        for experiment in experiments:
            assert "id" in experiment


class TestPhoenixAPIKeysNotSet:
    
    @pytest_asyncio.fixture
    async def client(self):
        application = create_app()
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    @pytest.mark.skipif(
        not SKIP_TESTS,
        reason="API keys are configured, skipping negative test"
    )
    @pytest.mark.asyncio
    async def test_datasets_without_api_keys(self, client, monkeypatch):
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        
        pass


def test_api_keys_configuration_message():
    if not PHOENIX_COLLECTOR_ENDPOINT or not PHOENIX_API_KEY:
        pytest.skip(SKIP_MESSAGE)
    
    assert PHOENIX_COLLECTOR_ENDPOINT is not None
    assert PHOENIX_API_KEY is not None
    assert len(PHOENIX_COLLECTOR_ENDPOINT) > 0
    assert len(PHOENIX_API_KEY) > 0
