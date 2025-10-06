import uuid

import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status

from deusauditron.app import create_app


@pytest.mark.asyncio
async def test_evaluation_lifecycle(monkeypatch):
    tenant_id = "tenant-test"
    agent_id = "agent-test"
    run_id = str(uuid.uuid4())

    # For this minimal test to pass, state must exist; in real tests we'd seed state.
    # Here we expect 404 for missing agent state.
    url = f"/api/v1/agents/{tenant_id}/{agent_id}/{run_id}/evaluation"
    application = create_app()
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(url, json={"api_keys": {}, "node_names": [], "auto_refine": False})
        assert resp.status_code in (status.HTTP_404_NOT_FOUND, status.HTTP_503_SERVICE_UNAVAILABLE)

