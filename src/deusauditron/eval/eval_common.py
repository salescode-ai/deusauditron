import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from deusauditron.llm_abstraction.llm_helper import LLMInvoker
from deusauditron.app_logging.logger import logger
from deusauditron.util.helper import Trinity
from deusauditron.schemas.autogen.references.eval_llm_response_schema import EvalLLMResponse
from deusauditron.schemas.autogen.references.model_schema import ModelConfig
from deusauditron.schemas.shared_models.models import LLMResponse, Message, MessageRole, MiniInteractionLog


@dataclass
class EvaluationContext:
    tenant_id: str
    agent_id: str
    run_id: str
    requested_node_names: Optional[List[str]]
    api_keys: Dict[str, str]
    auto_refine: bool


@dataclass
class ModelParams:
    model_name: str
    model_temperature: float
    reasoning: bool
    additional_params: Dict[str, Any]


@dataclass
class NodeBlock:
    node_name: str
    mini_interaction_logs: List[MiniInteractionLog]


@dataclass
class FailedRule:
    rule_name: str
    rule_instruction: str
    category: str
    failure_reasons: Dict[str, str]
    node_name: Optional[str] = None
    input_text: Optional[str] = None


@dataclass
class InterimEvaluationResult:
    rule_results: List[Any]
    failed_rules: List[FailedRule]
    additional_info: Optional[Dict[str, Any]] = None


class LLMResponseParser:
    @staticmethod
    async def parse_response(
        llm_response: LLMResponse, auto_refine_model_params: Optional[ModelParams] = None
    ) -> Optional[EvalLLMResponse]:
        if llm_response.error or not llm_response.content or llm_response.content.strip() == "":
            logger.error(
                f"Invalid LLM response: error={llm_response.error}, content='{llm_response.content}'"
            )
            return None
        processed_content = strip_markdown_formatting(llm_response.content)
        try:
            response_json = json.loads(processed_content)
            return EvalLLMResponse(**response_json)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to parse LLM response directly: {e}")
            return await LLMResponseParser._convert_response_to_json(llm_response, auto_refine_model_params)

    @staticmethod
    def _strip_markdown_formatting(content: str) -> str:
        return strip_markdown_formatting(content)

    @staticmethod
    async def _convert_response_to_json(
        llm_response: LLMResponse, auto_refine_model_params: Optional[ModelParams] = None
    ) -> Optional[EvalLLMResponse]:
        try:
            conversion_prompt_template = Trinity.get_llm_response_conversion_prompt()
            variables_dict = {"llm_response_content": llm_response.content}
            conversion_prompt = Trinity.replace_variables(conversion_prompt_template, variables_dict)
            llm_request = [Message(content=conversion_prompt, role=MessageRole.USER)]
            if auto_refine_model_params:
                conversion_response = await LLMInvoker.get_instance().invoke_non_streaming(
                    llm_request=llm_request,
                    model_name=auto_refine_model_params.model_name,
                    temperature=auto_refine_model_params.model_temperature,
                    reasoning=auto_refine_model_params.reasoning,
                    **auto_refine_model_params.additional_params,
                )
            else:
                conversion_response = await LLMInvoker.get_instance().invoke_non_streaming(
                    llm_request=llm_request,
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    reasoning=False,
                )
            if conversion_response.error or not conversion_response.content:
                return None
            try:
                converted_json = json.loads(conversion_response.content)
                return EvalLLMResponse(**converted_json)
            except (json.JSONDecodeError, ValidationError):
                processed_content = LLMResponseParser._strip_markdown_formatting(conversion_response.content)
                try:
                    converted_json = json.loads(processed_content)
                    return EvalLLMResponse(**converted_json)
                except (json.JSONDecodeError, ValidationError):
                    return None
        except Exception as e:
            logger.error(f"Error in response conversion: {e}")
            return None


class ModelParamsBuilder:
    @staticmethod
    async def build_model_params(model_list: List[ModelConfig], api_keys: Dict[str, str]) -> List[ModelParams]:
        model_params_list = []
        for model in model_list:
            additional_params = {}
            api_key = api_keys.get(model.name.split("/")[0].lower())
            if api_key:
                additional_params["api_key"] = api_key
            if model.reasoning_effort:
                additional_params["reasoning_effort"] = model.reasoning_effort.value
            model_params = ModelParams(
                model_name=model.name,
                model_temperature=model.temperature,
                reasoning=False,
                additional_params=additional_params,
            )
            model_params_list.append(model_params)
        return model_params_list


def strip_markdown_formatting(content: str) -> str:
    if not content:
        return content
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()

