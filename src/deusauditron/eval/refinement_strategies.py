import asyncio
import json
from typing import Any, Dict, List, Optional

from deusauditron.llm_abstraction.llm_helper import LLMInvoker
from deusauditron.app_logging.logger import logger
from deusauditron.util.helper import Trinity
from deusauditron.schemas.autogen.references.auto_refine_schema import AutoRefine, NodeRecommendation, IntentRecommendation
from deusauditron.schemas.autogen.references.llm_node_schema import LLMNodeConfig
from deusauditron.schemas.shared_models.models import LLMResponse, Message, MessageRole

from .eval_common import EvaluationContext, FailedRule, ModelParams, strip_markdown_formatting


class NodeLevelRefinementStrategy:
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.auto_refine_model_params = auto_refine_model_params

    async def refine_prompts(self, context: EvaluationContext, failed_node_rules: Dict[str, List[FailedRule]], failed_turn_rules: Dict[str, List[FailedRule]], nodes_under_evaluation: Dict[str, LLMNodeConfig]) -> list[NodeRecommendation]:
        logger.info("Starting node-level prompt refinement")
        union_failed_rules: Dict[str, List[FailedRule]] = {}
        for node_name, failed_rules in failed_node_rules.items():
            union_failed_rules[node_name] = failed_rules.copy()
        for node_name, failed_rules in failed_turn_rules.items():
            union_failed_rules.setdefault(node_name, []).extend(failed_rules)
        if not union_failed_rules:
            logger.info("No failed rules found for node-level refinement")
            return []
        tasks = [self._refine_node_prompt(context, node_name, failed_rules, nodes_under_evaluation) for node_name, failed_rules in union_failed_rules.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        recommendations: list[NodeRecommendation] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Node refinement failed: {result}")
                continue
            if result:
                node_name = list(union_failed_rules.keys())[i]
                recommendations.append(NodeRecommendation(node=node_name, prompt=str(result)))
        return recommendations

    async def _refine_node_prompt(self, context: EvaluationContext, node_name: str, failed_rules: List[FailedRule], nodes_under_evaluation: Dict[str, LLMNodeConfig]) -> Optional[str]:
        try:
            node = nodes_under_evaluation.get(node_name)
            if not node:
                return None
            node_system_prompt = node.system_prompt or ""
            if not node_system_prompt.strip():
                return None
            failed_rules_json = self._format_failed_rules_for_prompt(failed_rules)
            recommended_prompt = await self._call_auto_refine_llm(node_system_prompt, failed_rules_json)
            return recommended_prompt or None
        except Exception as e:
            logger.error(f"Error refining prompt for node '{node_name}': {e}")
            return None

    def _format_failed_rules_for_prompt(self, failed_rules: List[FailedRule]) -> str:
        formatted_rules: Dict[str, List[Dict[str, Any]]] = {}
        for failed_rule in failed_rules:
            node_name = failed_rule.node_name or "unknown"
            formatted_rules.setdefault(node_name, []).append({
                "rule_name": failed_rule.rule_name,
                "rule_instruction": failed_rule.rule_instruction,
                "failure_reasons": failed_rule.failure_reasons,
                "input_text": failed_rule.input_text,
            })
        return json.dumps(formatted_rules, indent=2)

    async def _call_auto_refine_llm(self, node_system_prompt: str, failed_rules_json: str) -> Optional[str]:
        if not self.auto_refine_model_params:
            return None
        prompt_template = Trinity.get_auto_refine_turn_node_prompt()
        prompt_content = prompt_template.replace("{node_system_prompt}", node_system_prompt).replace("{failed_rules}", failed_rules_json)
        llm_request = [Message(content=prompt_content, role=MessageRole.USER)]
        llm_response = await LLMInvoker.get_instance().invoke_non_streaming(
            llm_request=llm_request,
            model_name=self.auto_refine_model_params.model_name,
            temperature=self.auto_refine_model_params.model_temperature,
            reasoning=self.auto_refine_model_params.reasoning,
            **self.auto_refine_model_params.additional_params,
        )
        if llm_response.error or not llm_response.content:
            return None
        recommended_prompt = self._extract_recommended_prompt_from_json_response(llm_response.content)
        if recommended_prompt:
            return recommended_prompt
        return await self._convert_response_to_json(llm_response)

    def _extract_recommended_prompt_from_json_response(self, response_content: str) -> Optional[str]:
        try:
            processed_content = strip_markdown_formatting(response_content)
            response_json = json.loads(processed_content)
            if isinstance(response_json, dict) and "prompt" in response_json:
                prompt = response_json["prompt"]
                if isinstance(prompt, str) and prompt.strip():
                    return prompt.strip()
            return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    async def _convert_response_to_json(self, llm_response: LLMResponse) -> Optional[str]:
        if not self.auto_refine_model_params:
            return None
        conversion_prompt_template = Trinity.get_auto_refine_response_conversion_prompt()
        conversion_prompt = conversion_prompt_template.replace("{auto_refine_response_content}", llm_response.content)
        llm_request = [Message(content=conversion_prompt, role=MessageRole.USER)]
        conversion_response = await LLMInvoker.get_instance().invoke_non_streaming(
            llm_request=llm_request,
            model_name=self.auto_refine_model_params.model_name,
            temperature=self.auto_refine_model_params.model_temperature,
            reasoning=self.auto_refine_model_params.reasoning,
            **self.auto_refine_model_params.additional_params,
        )
        if conversion_response.error or not conversion_response.content:
            return None
        try:
            processed_content = strip_markdown_formatting(conversion_response.content)
            converted_json = json.loads(processed_content)
            if isinstance(converted_json, dict) and "prompt" in converted_json:
                prompt = converted_json["prompt"]
                if isinstance(prompt, str) and prompt.strip():
                    return prompt.strip()
            return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


class IntentLevelRefinementStrategy:
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.auto_refine_model_params = auto_refine_model_params

    async def refine_prompts(self, context: EvaluationContext, failed_intent_rules: Dict[str, List[FailedRule]], nodes_under_evaluation: Dict[str, LLMNodeConfig]) -> List[IntentRecommendation]:
        if not failed_intent_rules:
            return []
        from deusauditron.state.manager import StateManager
        state = None
        try:
            state = await StateManager().get_state(context.tenant_id, context.agent_id, context.run_id)
        except Exception:
            state = None
        intent_instructions = state.get_flow_data_intent_instructions() if state and hasattr(state, 'get_flow_data_intent_instructions') else {}
        recommendations: List[IntentRecommendation] = []
        for intent_node_name, failed_rules in failed_intent_rules.items():
            current_instructions = intent_instructions.get(intent_node_name, []) if intent_instructions else []
            refined_instructions = await self._refine_intent_instructions(intent_node_name, failed_rules, current_instructions)
            if refined_instructions:
                recommendations.append(IntentRecommendation(node=intent_node_name, intent_instructions=refined_instructions))
        return recommendations

    async def _refine_intent_instructions(self, intent_node_name: str, failed_rules: List[FailedRule], current_instructions: List[str]) -> Optional[Dict[str, str]]:
        try:
            failed_rules_json = self._format_failed_rules_for_prompt(failed_rules)
            intent_instructions_json = json.dumps(current_instructions, indent=2)
            refined_instructions = await self._call_auto_refine_intent_llm(intent_node_name, failed_rules_json, intent_instructions_json)
            return refined_instructions or None
        except Exception as e:
            logger.error(f"Error refining intent instructions for node '{intent_node_name}': {e}")
            return None

    def _format_failed_rules_for_prompt(self, failed_rules: List[FailedRule]) -> str:
        formatted_rules = []
        for failed_rule in failed_rules:
            formatted_rules.append({
                "rule_name": failed_rule.rule_name,
                "rule_instruction": failed_rule.rule_instruction,
                "failure_reasons": failed_rule.failure_reasons,
                "input_text": failed_rule.input_text,
            })
        return json.dumps(formatted_rules, indent=2)

    async def _call_auto_refine_intent_llm(self, intent_node_name: str, failed_rules_json: str, intent_instructions_json: str) -> Optional[Dict[str, str]]:
        if not self.auto_refine_model_params:
            return None
        prompt_template = Trinity.get_auto_refine_intent_prompt()
        prompt_content = prompt_template.replace("{failed_rules}", failed_rules_json).replace("{intent_instructions}", intent_instructions_json)
        llm_request = [Message(content=prompt_content, role=MessageRole.USER)]
        llm_response = await LLMInvoker.get_instance().invoke_non_streaming(
            llm_request=llm_request,
            model_name=self.auto_refine_model_params.model_name,
            temperature=self.auto_refine_model_params.model_temperature,
            reasoning=self.auto_refine_model_params.reasoning,
            **self.auto_refine_model_params.additional_params,
        )
        if llm_response.error or not llm_response.content:
            return None
        refined_instructions = self._extract_refined_instructions_from_json_response(llm_response.content)
        if refined_instructions:
            return refined_instructions
        return await self._convert_intent_response_to_json(llm_response)

    def _extract_refined_instructions_from_json_response(self, response_content: str) -> Optional[Dict[str, str]]:
        try:
            processed_content = strip_markdown_formatting(response_content)
            response_json = json.loads(processed_content)
            if isinstance(response_json, dict) and "intent_instructions" in response_json:
                instructions = response_json["intent_instructions"]
                if isinstance(instructions, dict):
                    return instructions
                if isinstance(instructions, str) and instructions.strip():
                    return {"intent": instructions.strip()}
                if isinstance(instructions, list) and instructions:
                    for instruction in instructions:
                        if isinstance(instruction, str) and instruction.strip():
                            return {"intent": instruction.strip()}
            return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    async def _convert_intent_response_to_json(self, llm_response: LLMResponse) -> Optional[Dict[str, str]]:
        if not self.auto_refine_model_params:
            return None
        conversion_prompt_template = Trinity.get_auto_refine_intent_response_conversion_prompt()
        conversion_prompt = conversion_prompt_template.replace("{auto_refine_intent_response_content}", llm_response.content)
        llm_request = [Message(content=conversion_prompt, role=MessageRole.USER)]
        conversion_response = await LLMInvoker.get_instance().invoke_non_streaming(
            llm_request=llm_request,
            model_name=self.auto_refine_model_params.model_name,
            temperature=self.auto_refine_model_params.model_temperature,
            reasoning=self.auto_refine_model_params.reasoning,
            **self.auto_refine_model_params.additional_params,
        )
        if conversion_response.error or not conversion_response.content:
            return None
        try:
            processed_content = strip_markdown_formatting(conversion_response.content)
            converted_json = json.loads(processed_content)
            if isinstance(converted_json, dict) and "intent_instructions" in converted_json:
                instructions = converted_json["intent_instructions"]
                if isinstance(instructions, dict):
                    return instructions
                if isinstance(instructions, str) and instructions.strip():
                    return {"intent": instructions.strip()}
                if isinstance(instructions, list) and instructions:
                    for instruction in instructions:
                        if isinstance(instruction, str) and instruction.strip():
                            return {"intent": instruction.strip()}
            return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


class ConversationLevelRefinementStrategy:
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.auto_refine_model_params = auto_refine_model_params

    async def refine_prompts(self, context: EvaluationContext, failed_conversation_rules: List[FailedRule], nodes_under_evaluation: Dict[str, LLMNodeConfig]) -> list[NodeRecommendation]:
        if not failed_conversation_rules:
            return []
        flow_rules = [rule for rule in failed_conversation_rules if rule.category == "conversation"]
        if not flow_rules:
            return []
        failed_rules_json = json.dumps([
            {"rule_name": r.rule_name, "rule_instruction": r.rule_instruction, "failure_reasons": r.failure_reasons, "input_text": r.input_text}
            for r in flow_rules
        ], indent=2)
        nodewise_system_prompts = json.dumps({name: {"node": name, "system_prompt": cfg.system_prompt or ""} for name, cfg in nodes_under_evaluation.items()}, indent=2)
        recommended_prompts = await self._call_auto_refine_flow_llm(failed_rules_json, nodewise_system_prompts)
        recommendations: list[NodeRecommendation] = []
        if recommended_prompts:
            for node_name, prompt in recommended_prompts.items():
                recommendations.append(NodeRecommendation(node=node_name, prompt=prompt))
        return recommendations

    async def _call_auto_refine_flow_llm(self, failed_rules_json: str, nodewise_system_prompts: str) -> Optional[Dict[str, str]]:
        if not self.auto_refine_model_params:
            return None
        prompt_template = Trinity.get_auto_refine_flow_prompt()
        prompt_content = prompt_template.replace("{failed_rules}", failed_rules_json).replace("{nodewise_system_prompts}", nodewise_system_prompts)
        llm_request = [Message(content=prompt_content, role=MessageRole.USER)]
        llm_response = await LLMInvoker.get_instance().invoke_non_streaming(
            llm_request=llm_request,
            model_name=self.auto_refine_model_params.model_name,
            temperature=self.auto_refine_model_params.model_temperature,
            reasoning=self.auto_refine_model_params.reasoning,
            **self.auto_refine_model_params.additional_params,
        )
        if llm_response.error or not llm_response.content:
            return None
        recommended_prompts = self._extract_recommended_prompts_from_json_response(llm_response.content)
        if recommended_prompts:
            return recommended_prompts
        return await self._convert_flow_response_to_json(llm_response)

    def _extract_recommended_prompts_from_json_response(self, response_content: str) -> Optional[Dict[str, str]]:
        try:
            processed_content = strip_markdown_formatting(response_content)
            response_json = json.loads(processed_content)
            if isinstance(response_json, list):
                prompts_dict: Dict[str, str] = {}
                for item in response_json:
                    if isinstance(item, dict) and "node" in item and "prompt" in item:
                        node_name = item["node"]
                        prompt = item["prompt"]
                        if isinstance(node_name, str) and isinstance(prompt, str) and prompt.strip():
                            prompts_dict[node_name] = prompt.strip()
                if prompts_dict:
                    return prompts_dict
            return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    async def _convert_flow_response_to_json(self, llm_response: LLMResponse) -> Optional[Dict[str, str]]:
        if not self.auto_refine_model_params:
            return None
        conversion_prompt_template = Trinity.get_auto_refine_flow_response_conversion_prompt()
        conversion_prompt = conversion_prompt_template.replace("{auto_refine_flow_response_content}", llm_response.content)
        llm_request = [Message(content=conversion_prompt, role=MessageRole.USER)]
        conversion_response = await LLMInvoker.get_instance().invoke_non_streaming(
            llm_request=llm_request,
            model_name=self.auto_refine_model_params.model_name,
            temperature=self.auto_refine_model_params.model_temperature,
            reasoning=self.auto_refine_model_params.reasoning,
            **self.auto_refine_model_params.additional_params,
        )
        if conversion_response.error or not conversion_response.content:
            return None
        try:
            processed_content = strip_markdown_formatting(conversion_response.content)
            converted_json = json.loads(processed_content)
            if isinstance(converted_json, list):
                prompts_dict: Dict[str, str] = {}
                for item in converted_json:
                    if isinstance(item, dict) and "node" in item and "prompt" in item:
                        node_name = item["node"]
                        prompt = item["prompt"]
                        if isinstance(node_name, str) and isinstance(prompt, str) and prompt.strip():
                            prompts_dict[node_name] = prompt.strip()
                if prompts_dict:
                    return prompts_dict
            return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


class AutoRefinementOrchestrator:
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.node_strategy = NodeLevelRefinementStrategy(auto_refine_model_params)
        self.intent_strategy = IntentLevelRefinementStrategy(auto_refine_model_params)
        self.conversation_strategy = ConversationLevelRefinementStrategy(auto_refine_model_params)

    async def perform_auto_refinement(self, context: EvaluationContext, failed_node_rules: Dict[str, List[FailedRule]], failed_turn_rules: Dict[str, List[FailedRule]], failed_intent_rules: Dict[str, List[FailedRule]], failed_conversation_rules: List[FailedRule], nodes_under_evaluation: Dict[str, LLMNodeConfig]) -> Optional[AutoRefine]:
        logger.info("Starting auto-refinement process")
        node_prompts = await self.node_strategy.refine_prompts(context, failed_node_rules, failed_turn_rules, nodes_under_evaluation)
        intent_prompts = await self.intent_strategy.refine_prompts(context, failed_intent_rules, nodes_under_evaluation)
        conversation_prompts = await self.conversation_strategy.refine_prompts(context, failed_conversation_rules, nodes_under_evaluation)
        validated_conversation_prompts = [p for p in conversation_prompts if p.node in nodes_under_evaluation]
        node_recommendations = node_prompts + validated_conversation_prompts
        intent_recommendations = intent_prompts
        return AutoRefine(node_recommendations=node_recommendations, global_recommendations="", intent_recommendations=intent_recommendations)

