from typing import Dict, List

from deusauditron.app_logging.logger import logger
from deusauditron.state.store.factory import get_state_store
from deusauditron.state.store.state_key import StateKey
from deusauditron.schemas.shared_models.models import InteractionLog, Message


class StateManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.store = get_state_store()

    async def set_eval_state(self, tenant_id: str, agent_id: str, run_id: str, state):
        logger.info(f"## StateManager.set_eval_state called for {tenant_id}/{agent_id}/{run_id}")
        await self.store.set_eval_state(StateKey(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id), state)
        logger.info(f"## Eval State successfully set for {tenant_id}/{agent_id}/{run_id}")

    async def get_eval_state(self, tenant_id: str, agent_id: str, run_id: str):
        return await self.store.get_eval_state(StateKey(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id))

    async def delete_eval_state(self, tenant_id: str, agent_id: str, run_id: str):
        await self.store.delete_eval_state(StateKey(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id))

    async def get_path(self, tenant_id: str, agent_id: str, run_id: str) -> List[str]:
        return await self.store.get_path(StateKey(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id))

    async def get_messages(self, tenant_id: str, agent_id: str, run_id: str) -> List[Message]:
        return await self.store.get_messages(StateKey(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id))

    async def get_interaction_log(self, tenant_id: str, agent_id: str, run_id: str) -> List[InteractionLog]:
        return await self.store.get_interaction_log(StateKey(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id))

    async def get_state(self, tenant_id: str, agent_id: str, run_id: str):
        return await getattr(self.store, "get_state")(StateKey(tenant_id=tenant_id, agent_id=agent_id, run_id=run_id))

