from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app_logging.logger import logger
from util.helper import Trinity
from schemas.autogen.references.eval_rule_schema import EvalRule, Type
from schemas.autogen.references.evaluation_rule_result_schema import EvaluationRuleResult
from schemas.shared_models.models import AIMessage, HumanMessage, InteractionLog, Message, MessageRole, MiniInteractionLog

from .eval_common import EvaluationContext, FailedRule, InterimEvaluationResult, ModelParams
from .eval_utils import EvaluationUtils


class IEvaluationStrategy(ABC):
    @abstractmethod
    async def evaluate(self, context: EvaluationContext, evaluation_data: Dict[str, Any], model_params_list: List[ModelParams]) -> InterimEvaluationResult:
        pass


class TurnEvaluationStrategy(IEvaluationStrategy):
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.eval_utils = EvaluationUtils(auto_refine_model_params)

    async def evaluate(self, context: EvaluationContext, evaluation_data: Dict[str, Any], model_params_list: List[ModelParams]) -> InterimEvaluationResult:
        node = evaluation_data["node"]
        interaction = evaluation_data["interaction"]
        message_history = evaluation_data["message_history"]
        if not node.evaluation_rules:
            return InterimEvaluationResult(rule_results=[], failed_rules=[])
        variables_dict = {
            "user_message": interaction.user_message,
            "llm_response": interaction.llm_response.raw_content,
            "message_history": Trinity.get_conversation_str(message_history),
        }
        eval_prompt_template = Trinity.get_turn_evaluation_prompt()
        eval_prompt_turn_level = Trinity.replace_variables(eval_prompt_template, variables_dict)
        results = []
        failed_rules = []
        for evaluation_rule in node.evaluation_rules:
            if evaluation_rule.type != Type.turn:
                continue
            variables_dict["evaluation_rule"] = evaluation_rule.instruction
            eval_prompt_rule_level = Trinity.replace_variables(eval_prompt_turn_level, variables_dict)
            llm_request = [Message(content=eval_prompt_rule_level, role=MessageRole.USER)]
            rule_result = EvaluationRuleResult(rule=evaluation_rule, model_results=[])
            model_results = await self.eval_utils.evaluate_with_models(llm_request, model_params_list)
            rule_result.model_results = model_results
            results.append(rule_result)
            failed_models = []
            failure_reasons = {}
            for model_result in model_results:
                if model_result.result == "FAIL":
                    failed_models.append(model_result.model_name)
                    failure_reasons[model_result.model_name] = model_result.reasoning or "No reasoning provided"
            if failed_models:
                failed_rule = FailedRule(
                    rule_name=evaluation_rule.name,
                    rule_instruction=evaluation_rule.instruction,
                    node_name=interaction.llm_response.node,
                    category="turn",
                    input_text=interaction.llm_response.raw_content,
                    failure_reasons=failure_reasons,
                )
                failed_rules.append(failed_rule)
        return InterimEvaluationResult(rule_results=results, failed_rules=failed_rules)


class NodeEvaluationStrategy(IEvaluationStrategy):
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.eval_utils = EvaluationUtils(auto_refine_model_params)

    async def evaluate(self, context: EvaluationContext, evaluation_data: Dict[str, Any], model_params_list: List[ModelParams]) -> InterimEvaluationResult:
        node = evaluation_data["node"]
        interaction_history = evaluation_data["interaction_history"]
        if not node.evaluation_rules or not interaction_history:
            return InterimEvaluationResult(rule_results=[], failed_rules=[])
        conversation_str = Trinity.get_interaction_str(interaction_history)
        variables_dict = {"conversation": conversation_str}
        results = []
        failed_rules = []
        for evaluation_rule in node.evaluation_rules:
            if evaluation_rule.type != Type.node:
                continue
            variables_dict["evaluation_rule"] = evaluation_rule.instruction
            eval_prompt_template = Trinity.get_node_evaluation_prompt()
            eval_prompt_node_level = Trinity.replace_variables(eval_prompt_template, variables_dict)
            llm_request = [Message(content=eval_prompt_node_level, role=MessageRole.USER)]
            rule_result = EvaluationRuleResult(rule=evaluation_rule, model_results=[])
            model_results = await self.eval_utils.evaluate_with_models(llm_request, model_params_list)
            rule_result.model_results = model_results
            results.append(rule_result)
            failed_models = []
            failure_reasons = {}
            for model_result in model_results:
                if model_result.result == "FAIL":
                    failed_models.append(model_result.model_name)
                    failure_reasons[model_result.model_name] = model_result.reasoning or "No reasoning provided"
            if failed_models:
                failed_rule = FailedRule(
                    rule_name=evaluation_rule.name,
                    rule_instruction=evaluation_rule.instruction,
                    node_name=(interaction_history[0].node if interaction_history else "unknown"),
                    category="node",
                    input_text=conversation_str,
                    failure_reasons=failure_reasons,
                )
                failed_rules.append(failed_rule)
        return InterimEvaluationResult(rule_results=results, failed_rules=failed_rules)


class IntentEvaluationStrategy(IEvaluationStrategy):
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.eval_utils = EvaluationUtils(auto_refine_model_params)

    async def evaluate(self, context: EvaluationContext, evaluation_data: Dict[str, Any], model_params_list: List[ModelParams]) -> InterimEvaluationResult:
        interaction = evaluation_data["interaction"]
        conversation_str = evaluation_data["conversation_str"]
        valid_intents = evaluation_data["valid_intents"]
        variables_dict = {
            "user_message": interaction.user_message,
            "detected_intent": interaction.llm_response.intent,
            "message_history": conversation_str,
            "valid_intents": valid_intents.get(interaction.llm_response.intent_detector, []),
        }
        intent_eval_prompt_template = Trinity.get_intent_evaluation_prompt()
        intent_eval_prompt = Trinity.replace_variables(intent_eval_prompt_template, variables_dict)
        llm_request = [Message(content=intent_eval_prompt, role=MessageRole.USER)]
        rule_name = f"Evaluate intent:{interaction.llm_response.intent or 'unknown'} detected by node:{interaction.llm_response.intent_detector or 'unknown'}"
        rule_result = EvaluationRuleResult(rule=EvalRule(name=rule_name, instruction="", type=Type.intent), model_results=[])
        model_results = await self.eval_utils.evaluate_with_models(llm_request, model_params_list)
        rule_result.model_results = model_results
        failed_rules = []
        failed_models = []
        failure_reasons = {}
        for model_result in model_results:
            if model_result.result == "FAIL":
                failed_models.append(model_result.model_name)
                failure_reasons[model_result.model_name] = model_result.reasoning or "No reasoning provided"
        if failed_models:
            failed_rule = FailedRule(
                rule_name=rule_name,
                rule_instruction="",
                node_name=interaction.llm_response.intent_detector or "unknown",
                category="intent",
                input_text=f"Intent: {interaction.llm_response.intent or 'unknown'}, Node: {interaction.llm_response.intent_detector or 'unknown'}",
                failure_reasons=failure_reasons,
            )
            failed_rules.append(failed_rule)
        return InterimEvaluationResult(rule_results=[rule_result], failed_rules=failed_rules)


class ConversationEvaluationStrategy(IEvaluationStrategy):
    def __init__(self, auto_refine_model_params: Optional[ModelParams] = None):
        self.eval_utils = EvaluationUtils(auto_refine_model_params)

    async def evaluate(self, context: EvaluationContext, evaluation_data: Dict[str, Any], model_params_list: List[ModelParams]) -> InterimEvaluationResult:
        messages = evaluation_data["messages"]
        conversation_eval_rules = evaluation_data["conversation_eval_rules"]
        if not conversation_eval_rules:
            return InterimEvaluationResult(rule_results=[], failed_rules=[])
        conversation_str = Trinity.get_conversation_str(messages)
        variables_dict = {"conversation": conversation_str}
        conversation_eval_prompt_template = Trinity.get_conversation_evaluation_prompt()
        results = []
        failed_rules = []
        for evaluation_rule in conversation_eval_rules:
            if evaluation_rule.type != Type.flow:
                continue
            variables_dict["evaluation_rule"] = evaluation_rule.instruction
            conversation_eval_prompt = Trinity.replace_variables(conversation_eval_prompt_template, variables_dict)
            llm_request = [Message(content=conversation_eval_prompt, role=MessageRole.USER)]
            rule_result = EvaluationRuleResult(rule=evaluation_rule, model_results=[])
            model_results = await self.eval_utils.evaluate_with_models(llm_request, model_params_list)
            rule_result.model_results = model_results
            results.append(rule_result)
            failed_models = []
            failure_reasons = {}
            for model_result in model_results:
                if model_result.result == "FAIL":
                    failed_models.append(model_result.model_name)
                    failure_reasons[model_result.model_name] = model_result.reasoning or "No reasoning provided"
            if failed_models:
                failed_rules.append(
                    FailedRule(
                        rule_name=evaluation_rule.name,
                        rule_instruction=evaluation_rule.instruction,
                        category="conversation",
                        input_text=conversation_str,
                        failure_reasons=failure_reasons,
                    )
                )
        return InterimEvaluationResult(rule_results=results, failed_rules=failed_rules, additional_info={"conversation_str": conversation_str})

