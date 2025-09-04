from typing import Dict, List

from state.object.eval import EvalState
from state.object.runtime import CurrentNode, InteractionLogList, MessagesList
from state.store.base import BaseStateStore
from state.store.state_key import StateKey
from schemas.shared_models.models import InteractionLog, Message


class LocalStateStore(BaseStateStore):
    _instance = None
    _eval_store: Dict[StateKey, EvalState]
    _path: Dict[StateKey, List[str]]
    _messages: Dict[StateKey, MessagesList]
    _interaction_log: Dict[StateKey, InteractionLogList]
    _node_output_dict: Dict[StateKey, Dict[str, str]]
    _current_node: Dict[StateKey, CurrentNode]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalStateStore, cls).__new__(cls)
            cls._instance._eval_store = {}
            cls._instance._path = {}
            cls._instance._messages = {}
            cls._instance._interaction_log = {}
            cls._instance._node_output_dict = {}
            cls._instance._current_node = {}
        return cls._instance

    async def set_state(self, key: StateKey, state):  # Not used by Deusauditron
        pass

    async def get_state(self, key: StateKey):  # Not used by Deusauditron
        return None

    async def delete_state(self, key: StateKey):  # Not used by Deusauditron
        pass

    async def clear_state(self):  # Not used by Deusauditron
        pass

    async def set_eval_state(self, key: StateKey, state: EvalState):
        self._eval_store[key] = state

    async def get_eval_state(self, key: StateKey) -> EvalState | None:
        return self._eval_store.get(key)

    async def delete_eval_state(self, key: StateKey):
        self._eval_store.pop(key, None)

    async def clear_eval_state(self):
        self._eval_store.clear()

    async def get_path(self, key: StateKey) -> List[str]:
        return self._path.get(key, [])

    async def set_path(self, key: StateKey, path: List[str]):
        self._path[key] = path

    async def get_messages(self, key: StateKey) -> List[Message]:
        messages_model = self._messages.get(key)
        if not messages_model:
            return []
        return messages_model.unwrap()

    async def set_messages(self, key: StateKey, messages: List[Message]):
        self._messages[key] = MessagesList.wrap(messages)

    async def get_interaction_log(self, key: StateKey) -> List[InteractionLog]:
        ilog_model = self._interaction_log.get(key)
        if not ilog_model:
            return []
        return ilog_model.unwrap()

    async def set_interaction_log(self, key: StateKey, interaction_log: List[InteractionLog]):
        self._interaction_log[key] = InteractionLogList.wrap(interaction_log)

    async def get_node_output(self, key: StateKey, node_name: str) -> str:
        node_output_dict = self._node_output_dict.get(key, {})
        return node_output_dict.get(node_name, "")

    async def get_all_node_outputs(self, key: StateKey) -> Dict[str, str]:
        return self._node_output_dict.get(key, {})

    async def set_node_output(self, key: StateKey, node_name: str, node_value: str) -> None:
        if node_name is None or node_value is None:
            return
        node_output_dict = self._node_output_dict.get(key, {})
        node_output_dict[node_name] = node_value

    async def get_current_node(self, key: StateKey) -> str:
        node_model = self._current_node.get(key)
        if not node_model:
            return ""
        node_name = node_model.unwrap()
        return node_name or ""

    async def set_current_node(self, key: StateKey, current_node: str):
        self._current_node[key] = CurrentNode.wrap(current_node)

    async def delete_agent(self, key: StateKey):
        for store_dict in [self._messages, self._node_output_dict, self._current_node]:
            store_dict.pop(key, None)

