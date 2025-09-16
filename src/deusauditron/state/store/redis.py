from typing import Dict, List, Optional, Tuple

import redis.asyncio as redis
from pydantic import TypeAdapter

from deusauditron.app_logging.logger import logger
from deusauditron.state.object.eval import EvalState
from deusauditron.state.object.runtime import CurrentNode, InteractionLogList, MessagesList, PathList
from deusauditron.state.store.base import BaseStateStore
from deusauditron.state.store.state_key import StateKey
from deusauditron.schemas.shared_models.models import InteractionLog, Message
from deusauditron.state.object.base import BaseState

# Minimal adapters: we are not using a discriminated union here; simple adapters suffice
agent_state_adapter = TypeAdapter(BaseState)
eval_state_adapter = TypeAdapter(EvalState)


class RedisStateStore(BaseStateStore):
    def __init__(self, redis_url: str, prefix: str = "state"):
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix

    def _key_eval_state(self, key: StateKey) -> str:
        return f"{{{self.prefix}:{key.tenant_id}}}:{key.agent_id}:{key.run_id}:eval"

    def _key_path(self, key: StateKey) -> str:
        return f"{{{self.prefix}:{key.tenant_id}}}:{key.agent_id}:{key.run_id}:path"

    def _key_messages(self, key: StateKey) -> str:
        return f"{{{self.prefix}:{key.tenant_id}}}:{key.agent_id}:{key.run_id}:messages"

    def _key_interaction_log(self, key: StateKey) -> str:
        return f"{{{self.prefix}:{key.tenant_id}}}:{key.agent_id}:{key.run_id}:interaction_log"

    def _key_node_output(self, key: StateKey) -> str:
        return f"{{{self.prefix}:{key.tenant_id}}}:{key.agent_id}:{key.run_id}:node_outputs"

    def _key_current_node(self, key: StateKey) -> str:
        return f"{{{self.prefix}:{key.tenant_id}}}:{key.agent_id}:{key.run_id}:current_node"

    def _key_static_state(self, key: StateKey) -> str:
        return f"{{{self.prefix}:{key.tenant_id}}}:{key.agent_id}:{key.run_id}:static"

    async def set_state(self, key: StateKey, state):
        pass

    async def get_state(self, key: StateKey) -> Optional[BaseState]:
        redis_key = self._key_static_state(key)
        value = await self.client.get(redis_key)
        if value is None:
            return None
        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            state = agent_state_adapter.validate_json(value)
            return state
        except Exception as e:
            logger.error(f"Error deserializing state: {e}")
            return None

    async def delete_state(self, key: StateKey):
        pass

    async def clear_state(self):
        pass

    async def set_eval_state(self, key: StateKey, state: EvalState):
        redis_key = self._key_eval_state(key)
        json_data = eval_state_adapter.dump_json(state, by_alias=True)
        await self.client.set(redis_key, json_data)

    async def get_eval_state(self, key: StateKey) -> Optional[EvalState]:
        redis_key = self._key_eval_state(key)
        value = await self.client.get(redis_key)
        if value is None:
            return None
        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            state = eval_state_adapter.validate_json(value)
            return state
        except Exception as e:
            logger.error(f"Error deserializing eval state: {e}")
            return None

    async def delete_eval_state(self, key: StateKey):
        await self.client.delete(self._key_eval_state(key))

    async def clear_eval_state(self):
        pattern = f"{{{self.prefix}:*}}:*:eval"
        keys = await self.client.keys(pattern)
        if keys:
            await self.client.delete(*keys)

    async def get_path(self, key: StateKey) -> List[str]:
        raw = await self.client.get(self._key_path(key))
        if raw is None:
            return []
        return PathList.model_validate_json(raw).unwrap()

    async def set_path(self, key: StateKey, path: List[str]):
        await self.client.set(self._key_path(key), PathList.wrap(path).model_dump_json())

    async def set_messages(self, key: StateKey, messages: List[Message]):
        await self.client.set(self._key_messages(key), MessagesList.wrap(messages).model_dump_json())

    async def get_messages(self, key: StateKey) -> List[Message]:
        raw = await self.client.get(self._key_messages(key))
        if raw is None:
            return []
        return MessagesList.model_validate_json(raw).unwrap()

    async def set_interaction_log(self, key: StateKey, interaction_log: List[InteractionLog]):
        await self.client.set(self._key_interaction_log(key), InteractionLogList.wrap(interaction_log).model_dump_json())

    async def get_interaction_log(self, key: StateKey) -> List[InteractionLog]:
        raw = await self.client.get(self._key_interaction_log(key))
        if raw is None:
            return []
        return InteractionLogList.model_validate_json(raw).unwrap()

    async def get_node_output(self, key: StateKey, node_name: str) -> str:
        node_value = await self.client.hget(self._key_node_output(key), node_name)  # type: ignore
        return node_value or ""

    async def get_all_node_outputs(self, key: StateKey) -> Dict[str, str]:
        return await self.client.hgetall(self._key_node_output(key))  # type: ignore

    async def set_node_output(self, key: StateKey, node_name: str, node_value: str) -> None:
        if node_name is None or node_value is None:
            return
        await self.client.hset(self._key_node_output(key), node_name, node_value)  # type: ignore

    async def get_current_node(self, key: StateKey) -> str:
        raw = await self.client.get(self._key_current_node(key))
        if raw is None:
            return ""
        try:
            return CurrentNode.model_validate_json(raw).unwrap() or ""
        except Exception:
            try:
                return CurrentNode.model_validate_json(raw.decode("utf-8")).unwrap() or ""
            except Exception:
                return ""

    async def set_current_node(self, key: StateKey, current_node: str) -> None:
        await self.client.set(self._key_current_node(key), CurrentNode.wrap(current_node).model_dump_json())

    async def delete_agent(self, key: StateKey):
        pattern = f"{{{self.prefix}:{key.tenant_id}}}:{key.agent_id}:{key.run_id}:*"
        keys = await self.client.keys(pattern)
        if keys:
            await self.client.delete(*keys)

