import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from config import get_config
from app_logging.logger import logger
from schemas.shared_models.models import LLMResponse, Message


class LLMInvoker:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMInvoker, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_retries: int = 3, llm_adapter: Optional[Any] = None):
        if not self._initialized:
            config = get_config()
            self.max_retries = config.llm.max_retries
            self.llm_adapter = llm_adapter
            self._initialized = True
            self.fallback_model = config.llm.fallback_model
            logger.info("LLMInvoker initialized with adapter")

    @classmethod
    def get_instance(cls) -> "LLMInvoker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _handle_retry_error(self, e: Exception, current_retry_attempt: int, model_name: str) -> str:
        logger.warning(
            f"Call attempt {current_retry_attempt}/{self.max_retries} failed. Error: {str(e)}"
        )
        if current_retry_attempt == 1:
            model_name = self.fallback_model
            logger.info(f"Switching to fallback model: {model_name}")
        if current_retry_attempt == self.max_retries:
            final_err_msg = (
                f"LLM call failed after {current_retry_attempt}/{self.max_retries} attempts. Final error: {str(e)}"
            )
            logger.error(final_err_msg)
            raise RuntimeError(final_err_msg)
        return model_name

    def _update_response_format(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        response_schema = kwargs.pop("response_schema", None)
        if response_schema is not None:
            try:
                schema_dict = json.loads(response_schema) if isinstance(response_schema, str) else dict(response_schema)
            except Exception:
                schema_dict = {}
            response_format_dict = {"type": "json_schema", "json_schema": {"name": "response_generic_schema", "schema": schema_dict}}
            logger.info(f"Response format JSON: {response_format_dict}")
            kwargs["response_format"] = response_format_dict
        return kwargs

    async def invoke_non_streaming(
        self,
        llm_request: List[Message],
        model_name: str,
        temperature: float,
        reasoning: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        if not self.llm_adapter:
            raise RuntimeError("LLM adapter not initialized")
        kwargs = self._update_response_format(kwargs)
        for current_retry_attempt in range(1, self.max_retries + 1):
            try:
                return await self.llm_adapter.invoke(
                    messages=llm_request,
                    model_name=model_name,
                    temperature=temperature,
                    reasoning=reasoning,
                    **kwargs,
                )
            except Exception as e:
                model_name = self._handle_retry_error(e, current_retry_attempt, model_name)
        raise RuntimeError("LLM call failed after all retries")

    async def invoke_streaming(
        self,
        llm_request: List[Message],
        model_name: str,
        temperature: float,
        reasoning: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        if not self.llm_adapter:
            raise RuntimeError("LLM adapter not initialized")
        kwargs = self._update_response_format(kwargs)
        for current_retry_attempt in range(1, self.max_retries + 1):
            try:
                stream_gen = self.llm_adapter.stream(
                    messages=llm_request,
                    model_name=model_name,
                    temperature=temperature,
                    reasoning=reasoning,
                    **kwargs,
                )
                async for chunk in stream_gen:  # type: ignore[attr-defined]
                    yield chunk
                return
            except Exception as e:
                model_name = self._handle_retry_error(e, current_retry_attempt, model_name)

