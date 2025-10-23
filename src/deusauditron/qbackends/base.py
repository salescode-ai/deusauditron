from abc import ABC, abstractmethod
from typing import Optional

from deusauditron.schemas.shared_models.models import AgentEvalRequest, AgentRunRequest, VoiceEvalRequest


class BaseQueueBackend(ABC):
    @abstractmethod
    async def enqueue_run_request(self, request: AgentRunRequest) -> None:
        pass

    @abstractmethod
    async def dequeue_run_request(self) -> Optional[AgentRunRequest]:
        pass

    @abstractmethod
    async def enqueue_eval_request(self, request: AgentEvalRequest) -> None:
        pass

    @abstractmethod
    async def dequeue_eval_request(self) -> Optional[AgentEvalRequest]:
        pass

    @abstractmethod
    async def enqueue_voice_eval_request(self, request: VoiceEvalRequest) -> None:
        pass

    @abstractmethod
    async def dequeue_voice_eval_request(self) -> Optional[VoiceEvalRequest]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

