import json
import re
import asyncio
from typing import List, Optional, Dict, Any

from llm_abstraction.llm_helper import LLMInvoker
from app_logging.logger import logger
from schemas.autogen.references.eval_llm_response_schema import EvalLLMResponse
from schemas.autogen.references.evaluation_rule_result_schema import ModelResult
from schemas.shared_models.models import LLMResponse, Message

from .eval_common import LLMResponseParser, ModelParams
from deusmachine_adapter.dm_adapter import DMAdapter


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

    async def custom_evaluator(self, final_output: str, expected_output: str, dm_adapter: DMAdapter) -> Dict[str, Any]:
        user_input = f"""You are an expert AI evaluator tasked with determining if two outputs match in terms of meaning, intent, and content quality. You will compare a "Final Output" (actual result) with an "Expected Output" (desired result) and determine if they are equivalent.

**Instructions:**
1. Analyze both outputs for semantic similarity, not just literal text matching
2. Consider if the final output fulfills the same intent and purpose as the expected output
3. Look for key information, facts, conclusions, and overall messaging alignment
4. Minor differences in wording, formatting, or style should not cause a failure if the core content and intent match
5. However, factual inaccuracies, missing key information, or fundamentally different conclusions should result in failure
6. Consider context appropriateness and completeness of the response
7. If the final output contains extra information that is not present in the expected output, it should not be considered as a failure. You have to match whether the final output contains the information that is present in the expected output.

**Expected Output:**
```
{expected_output}
```

**Final Output (to be evaluated):**
```
{final_output}
```

**Evaluation Criteria:**
- Content Accuracy: Does the final output contain the correct information?
- Intent Fulfillment: Does the final output serve the same purpose as the expected output?
- Completeness: Are all key points from the expected output addressed?
- Factual Consistency: Are there any contradictions or inaccuracies?
- Contextual Appropriateness: Is the response suitable for the intended context?

**Return Format:**
You must return your response wrapped in markdown code blocks with the JSON inside. Use this exact format:

```json
{{
    "result": "PASS" or "FAIL",
    "reasoning": "Detailed explanation of why the evaluation passed or failed, including specific points of comparison"
}}
```

**Examples of Pass Scenarios:**
- Same information presented in different words
- Additional helpful context that doesn't contradict the expected output
- Minor formatting or stylistic differences
- Equivalent conclusions reached through different explanations

**Examples of Fail Scenarios:**
- Missing critical information from expected output
- Factually incorrect statements
- Contradictory conclusions
- Incomplete responses that don't address the core requirements
- Responses that serve a fundamentally different purpose

Evaluate the outputs now and return your JSON response wrapped in code blocks."""

        model = {
            "name": "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
            "temperature": 0.1,
        }

        response = await dm_adapter.completions_api(user_input=user_input, model=model)
        try:
            if isinstance(response, dict):
                response_content = response.get("content", str(response))
            else:
                response_content = str(response)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                evaluation_result = json.loads(json_content)
                return {
                    "output": final_output,
                    "expected_output": expected_output,
                    "result": evaluation_result.get("result", "UNDEFINED"),
                    "reasoning": evaluation_result.get("reasoning", f"No JSON code block found in response: {response_content}"),
                }
            else:
                return {
                    "output": final_output,
                    "expected_output": expected_output,
                    "result": "UNDEFINED",
                    "reasoning": f"No JSON code block found in response: {response_content}",
                }
                
        except Exception as e:
            logger.error(f"Error parsing custom evaluator response: {e}")
            return {
                "output": final_output,
                "expected_output": expected_output,
                "result": "UNDEFINED",
                "reasoning": f"Error processing evaluation: {str(e)}"
            }
