import uuid
from typing import Any, AsyncGenerator, Dict, List

import litellm
from openinference.instrumentation import using_session

from deusauditron.config import TracingManager
from deusauditron.app_logging.logger import logger
from deusauditron.schemas.shared_models.models import LLMResponse, Message


class LiteLLMAdapter:
    def _convert_messages_to_json(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def convert_from_provider_response(self, provider_response: Any, stream_chunk: bool = False, reasoning: bool = False) -> LLMResponse:
        content = ""
        if provider_response is None:
            return LLMResponse(content="")
        reasoning_content = ""
        usage = ""
        if stream_chunk:
            if provider_response.choices and provider_response.choices[0].delta:
                content = provider_response.choices[0].delta.content or ""
                if reasoning and hasattr(provider_response.choices[0].delta, "reasoning_content"):
                    reasoning_content = provider_response.choices[0].delta.reasoning_content or ""
        else:
            if provider_response.choices and provider_response.choices[0].message:
                content = provider_response.choices[0].message.content or ""
                if reasoning and hasattr(provider_response.choices[0].message, "reasoning_content"):
                    reasoning_content = provider_response.choices[0].message.reasoning_content or ""
            if hasattr(provider_response, "usage") and provider_response.usage:
                usage = str(provider_response.usage)
        return LLMResponse(content=content, reasoning_content=reasoning_content, usage=usage)

    async def invoke(self, messages: List[Message], model_name: str, temperature: float, reasoning: bool = False, **kwargs: Any) -> LLMResponse:
        try:
            messages_json = self._convert_messages_to_json(messages)
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ["tenant_id", "agent_id", "run_id"]}
            if TracingManager().is_enabled:
                session_id = f"{kwargs.get('tenant_id', 'notenant')}/{kwargs.get('agent_id', 'noagent')}/{kwargs.get('run_id', str(uuid.uuid4()))}"
                with using_session(session_id):
                    resp = await litellm.acompletion(model=model_name, messages=messages_json, temperature=temperature, stream=False, drop_params=True, **clean_kwargs)
            else:
                resp = await litellm.acompletion(model=model_name, messages=messages_json, temperature=temperature, stream=False, drop_params=True, **clean_kwargs)
            return self.convert_from_provider_response(resp, stream_chunk=False, reasoning=reasoning)
        except Exception as e:
            logger.error(f"LiteLLM acompletion error: {e}")
            return LLMResponse(content="", error=True, error_text=str(e))

    async def stream(self, messages: List[Message], model_name: str, temperature: float, reasoning: bool = False, **kwargs: Any) -> AsyncGenerator[LLMResponse, None]:
        try:
            messages_json = self._convert_messages_to_json(messages)
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ["tenant_id", "agent_id", "run_id"]}
            if TracingManager().is_enabled:
                session_id = f"{kwargs.get('tenant_id', 'notenant')}/{kwargs.get('agent_id', 'noagent')}/{kwargs.get('run_id', str(uuid.uuid4()))}"
                with using_session(session_id):
                    async_gen = await litellm.acompletion(model=model_name, messages=messages_json, temperature=temperature, stream=True, drop_params=True, **clean_kwargs)
            else:
                async_gen = await litellm.acompletion(model=model_name, messages=messages_json, temperature=temperature, stream=True, drop_params=True, **clean_kwargs)

            # async_gen is an async iterator of chunks; iterate it directly
            async for chunk in async_gen:  # type: ignore
                yield self.convert_from_provider_response(chunk, stream_chunk=True, reasoning=reasoning)
        except Exception as e:
            logger.error(f"LiteLLM streaming error: {e}")
            yield LLMResponse(content="", error=True, error_text=str(e))

