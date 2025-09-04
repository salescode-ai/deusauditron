from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from typing import Any
BaseState = Any  # minimal typing; deusauditron does not manage full BaseState
from state.object.eval import EvalState
from schemas.shared_models.models import InteractionLog, Message
from .state_key import StateKey


class BaseStateStore(ABC):
    @abstractmethod
    async def set_state(self, key: StateKey, state: BaseState):
        pass

    @abstractmethod
    async def get_state(self, key: StateKey) -> Optional[BaseState]:
        pass

    @abstractmethod
    async def delete_state(self, key: StateKey):
        pass

    @abstractmethod
    async def clear_state(self):
        pass

    @abstractmethod
    async def set_eval_state(self, key: StateKey, state: EvalState):
        pass

    @abstractmethod
    async def get_eval_state(self, key: StateKey) -> Optional[EvalState]:
        pass

    @abstractmethod
    async def delete_eval_state(self, key: StateKey):
        pass

    @abstractmethod
    async def clear_eval_state(self):
        pass

    @abstractmethod
    async def get_path(self, key: StateKey) -> List[str]:
        pass

    @abstractmethod
    async def set_path(self, key: StateKey, path: List[str]):
        pass

    @abstractmethod
    async def get_messages(self, key: StateKey) -> List[Message]:
        pass

    @abstractmethod
    async def set_messages(self, key: StateKey, messages: List[Message]):
        pass

    @abstractmethod
    async def get_interaction_log(self, key: StateKey) -> List[InteractionLog]:
        pass

    @abstractmethod
    async def set_interaction_log(self, key: StateKey, interaction_log: List[InteractionLog]):
        pass

    @abstractmethod
    async def get_node_output(self, key: StateKey, node_name: str) -> str:
        pass

    @abstractmethod
    async def get_all_node_outputs(self, key: StateKey) -> Dict[str, str]:
        pass

    @abstractmethod
    async def set_node_output(self, key: StateKey, node_name: str, node_value: str):
        pass

    @abstractmethod
    async def get_current_node(self, key: StateKey) -> str:
        pass

    @abstractmethod
    async def set_current_node(self, key: StateKey, current_node: str):
        pass

    @abstractmethod
    async def delete_agent(self, key: StateKey):
        pass

