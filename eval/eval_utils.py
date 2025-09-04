import asyncio
from typing import List, Optional

from llm_abstraction.llm_helper import LLMInvoker
from app_logging.logger import logger
from schemas.autogen.references.eval_llm_response_schema import EvalLLMResponse
from schemas.autogen.references.evaluation_rule_result_schema import ModelResult
from schemas.shared_models.models import LLMResponse, Message

from .eval_common import LLMResponseParser, ModelParams


class EvaluationUtils:
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.auto_refine_model_params = auto_refine_model_params

    async def evaluate_with_models(self, llm_request: List[Message], model_params_list: List[ModelParams]) -> List[ModelResult]:
        tasks = [self._single_model_evaluation(llm_request, mp) for mp in model_params_list]
        raw_results: List[object] = await asyncio.gather(*tasks, return_exceptions=True)
        model_results: List[ModelResult] = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                logger.error(f"Model evaluation failed: {res}")
                model_results.append(ModelResult(model_name=model_params_list[i].model_name, result="ERROR", reasoning=f"Evaluation failed: {str(res)}"))
            else:
                # Safe cast since we know our task returns ModelResult on success
                model_results.append(res)  # type: ignore[arg-type]
        return model_results

    async def _single_model_evaluation(self, llm_request: List[Message], model_params: ModelParams) -> ModelResult:
        try:
            llm_response = await LLMInvoker.get_instance().invoke_non_streaming(
                llm_request,
                model_params.model_name,
                model_params.model_temperature,
                model_params.reasoning,
                **model_params.additional_params,
            )
            eval_llm_response = await LLMResponseParser.parse_response(llm_response, self.auto_refine_model_params)
            return self._build_model_result(model_params.model_name, eval_llm_response)
        except Exception as e:
            logger.error(f"Single model evaluation failed for {model_params.model_name}: {e}")
            return ModelResult(model_name=model_params.model_name, result="ERROR", reasoning=f"Model evaluation failed: {str(e)}")

    def _build_model_result(self, model_name: str, eval_llm_response: Optional[EvalLLMResponse]) -> ModelResult:
        if eval_llm_response is None:
            return ModelResult(model_name=model_name, result="NO_EVAL", reasoning="Model did not return a valid response")
        return ModelResult(model_name=model_name, result=eval_llm_response.result.value, reasoning=eval_llm_response.reason)

