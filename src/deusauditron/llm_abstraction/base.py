from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, List

from deusauditron.schemas.shared_models.models import LLMResponse, Message


class LLMInterface(ABC):
    @abstractmethod
    async def invoke(
        self,
        messages: List[Message],
        model_name: str,
        temperature: float,
        reasoning: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        model_name: str,
        temperature: float,
        reasoning: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        if False:
            yield LLMResponse(content="")  # type: ignore[misc]
        raise NotImplementedError

