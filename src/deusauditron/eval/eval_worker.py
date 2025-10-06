import asyncio
import copy
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from deusauditron.app_logging.context import set_logging_context
from deusauditron.app_logging.logger import logger
from deusauditron.config import TracingManager
from deusauditron.state.manager import StateManager
from deusauditron.util.helper import Trinity
from deusauditron.schemas.autogen.references.evaluation_result_schema import EvaluationResult, Status
from deusauditron.schemas.autogen.references.granular_evaluation_result_schema import Category, GranularEvaluationResults
from deusauditron.schemas.autogen.references.llm_node_schema import LLMNodeConfig
from deusauditron.schemas.shared_models.models import AgentEvalRequest, AIMessage, HumanMessage, InteractionLog, Message, MiniInteractionLog

from .eval_common import EvaluationContext, FailedRule, ModelParams, ModelParamsBuilder, NodeBlock
from .eval_state_manager import EvalStateManager
from .eval_strategies import ConversationEvaluationStrategy, IntentEvaluationStrategy, NodeEvaluationStrategy, TurnEvaluationStrategy
from .progress_handler import EvaluationProgressHandler
from .progress_tracker import MilestoneType
from .refinement_strategies import AutoRefinementOrchestrator


class EvaluationError(Exception):
    pass


class FailedRulesContainer:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._failed_node_rules: Dict[str, List[FailedRule]] = {}
        self._failed_turn_rules: Dict[str, List[FailedRule]] = {}
        self._failed_intent_rules: Dict[str, List[FailedRule]] = {}
        self._failed_conversation_rules: List[FailedRule] = []

    async def add_failed_node_rule(self, node_name: str, failed_rule: FailedRule) -> None:
        async with self._lock:
            self._failed_node_rules.setdefault(node_name, []).append(failed_rule)

    async def add_failed_turn_rule(self, node_name: str, failed_rule: FailedRule) -> None:
        async with self._lock:
            self._failed_turn_rules.setdefault(node_name, []).append(failed_rule)

    async def add_failed_intent_rule(self, node_name: str, failed_rule: FailedRule) -> None:
        async with self._lock:
            self._failed_intent_rules.setdefault(node_name, []).append(failed_rule)

    async def add_failed_conversation_rule(self, failed_rule: FailedRule) -> None:
        async with self._lock:
            self._failed_conversation_rules.append(failed_rule)

    async def get_failed_node_rules(self) -> Dict[str, List[FailedRule]]:
        async with self._lock:
            return copy.deepcopy(self._failed_node_rules)

    async def get_failed_turn_rules(self) -> Dict[str, List[FailedRule]]:
        async with self._lock:
            return copy.deepcopy(self._failed_turn_rules)

    async def get_failed_intent_rules(self) -> Dict[str, List[FailedRule]]:
        async with self._lock:
            return copy.deepcopy(self._failed_intent_rules)

    async def get_failed_conversation_rules(self) -> List[FailedRule]:
        async with self._lock:
            return copy.deepcopy(self._failed_conversation_rules)


class LLMEvaluator:
    def __init__(self, request: AgentEvalRequest):
        self.context = EvaluationContext(
            tenant_id=request.tenant_id,
            agent_id=request.agent_id,
            run_id=request.run_id,
            requested_node_names=(request.payload.node_names if request.payload else None),
            api_keys={},
            auto_refine=request.payload.auto_refine if request.payload else False,
        )
        self.request = request
        self.nodes_under_evaluation: Dict[str, LLMNodeConfig] = {}
        self.eval_state_manager = EvalStateManager(self.context.tenant_id, self.context.agent_id, self.context.run_id)
        self.progress_handler = EvaluationProgressHandler(self.eval_state_manager, auto_refine_enabled=self.context.auto_refine)
        self.turn_eval_model_params: List[ModelParams] = []
        self.node_eval_model_params: List[ModelParams] = []
        self.intent_eval_model_params: List[ModelParams] = []
        self.conversation_eval_model_params: List[ModelParams] = []
        self.auto_refine_model_params: Optional[ModelParams] = None
        self.start_time: Optional[datetime] = None
        self.failed_rules = FailedRulesContainer()
        self.eval_result: Optional[EvaluationResult] = None
        set_logging_context(self.context.tenant_id, self.context.agent_id, self.context.run_id)

    async def evaluate(self) -> None:
        try:
            logger.info("Starting evaluation process")
            await self._load_data_and_prepare()
            await self._execute_evaluations_parallel()
            if self.context.auto_refine:
                await self._auto_refine_evaluations()
            await self._update_final_state()
            logger.info("Evaluation process completed successfully")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            await self._update_error_state()
            raise EvaluationError(f"Evaluation failed: {str(e)}") from e

    async def _load_data_and_prepare(self) -> None:
        state = None
        try:
            state = await StateManager().get_state(self.context.tenant_id, self.context.agent_id, self.context.run_id)
        except Exception:
            state = None
        if not state:
            raise EvaluationError(f"State not found for {self.context.tenant_id}/{self.context.agent_id}/{self.context.run_id}")
        eval_state = await StateManager().get_eval_state(self.context.tenant_id, self.context.agent_id, self.context.run_id)
        if not eval_state:
            raise EvaluationError(f"Eval state not found for {self.context.tenant_id}/{self.context.agent_id}/{self.context.run_id}")
        if hasattr(state, "api_keys"):
            self.context.api_keys = state.api_keys.copy()
        if self.request.payload and self.request.payload.api_keys:
            self.context.api_keys.update(self.request.payload.api_keys)
        await self._find_eligible_nodes(state)
        await self._prepare_model_params(state)
        self.start_time = datetime.now(timezone.utc)
        await self.eval_state_manager.update_start_state(start_time=self.start_time)

    async def _find_eligible_nodes(self, state) -> None:
        if self.context.requested_node_names:
            eval_node_names = list(dict.fromkeys(self.context.requested_node_names))
        else:
            path = await StateManager().get_path(self.context.tenant_id, self.context.agent_id, self.context.run_id)
            interaction_log = await StateManager().get_interaction_log(self.context.tenant_id, self.context.agent_id, self.context.run_id)
            eval_node_names_set = set(path)
            for interaction in interaction_log:
                if interaction.llm_response.node:
                    eval_node_names_set.add(interaction.llm_response.node)
                if interaction.llm_response.intent_detector:
                    eval_node_names_set.add(interaction.llm_response.intent_detector)
            eval_node_names = list(eval_node_names_set)
        logger.info(f"Nodes eligible for evaluation: {eval_node_names}")
        for node_name in eval_node_names:
            node = state.get_flow_data_llm_nodes().get(node_name) if hasattr(state, 'get_flow_data_llm_nodes') else None
            if not node and hasattr(state, 'get_global_llm_nodes'):
                node = state.get_global_llm_nodes().get(node_name)
            if node:
                self.nodes_under_evaluation[node_name] = node
        logger.info(f"Successfully loaded {len(self.nodes_under_evaluation)} nodes for evaluation")

    async def _prepare_model_params(self, state) -> None:
        eval_config = state.get_evaluation_config() if state and hasattr(state, 'get_evaluation_config') else None
        if not eval_config:
            raise EvaluationError("Evaluation config not found")
        self.turn_eval_model_params = await ModelParamsBuilder.build_model_params(eval_config.turn_eval_models, self.context.api_keys)
        self.node_eval_model_params = await ModelParamsBuilder.build_model_params(eval_config.node_eval_models, self.context.api_keys)
        self.intent_eval_model_params = await ModelParamsBuilder.build_model_params(eval_config.intent_eval_models, self.context.api_keys)
        self.conversation_eval_model_params = await ModelParamsBuilder.build_model_params(eval_config.conversation_eval_models, self.context.api_keys)
        self.auto_refine_model_params = (await ModelParamsBuilder.build_model_params([eval_config.auto_refine_model], self.context.api_keys))[0]

    async def _auto_refine_evaluations(self) -> None:
        logger.info("Starting auto-refinement of failed evaluations")
        failed_node_rules = await self.failed_rules.get_failed_node_rules()
        failed_turn_rules = await self.failed_rules.get_failed_turn_rules()
        failed_intent_rules = await self.failed_rules.get_failed_intent_rules()
        failed_conversation_rules = await self.failed_rules.get_failed_conversation_rules()
        orchestrator = AutoRefinementOrchestrator(self.auto_refine_model_params)
        auto_refine = await orchestrator.perform_auto_refinement(
            context=self.context,
            failed_node_rules=failed_node_rules,
            failed_turn_rules=failed_turn_rules,
            failed_intent_rules=failed_intent_rules,
            failed_conversation_rules=failed_conversation_rules,
            nodes_under_evaluation=self.nodes_under_evaluation,
        )
        if auto_refine:
            await self.eval_state_manager.update_auto_refinements([auto_refine])
            if self.eval_result is not None:
                self.eval_result.auto_refinements = [auto_refine]
        await self.progress_handler.mark_milestone_complete(MilestoneType.AUTO_REFINE_EVALUATIONS)

    async def _execute_evaluations_parallel(self) -> None:
        interaction_log = await StateManager().get_interaction_log(self.context.tenant_id, self.context.agent_id, self.context.run_id)
        messages = await StateManager().get_messages(self.context.tenant_id, self.context.agent_id, self.context.run_id)
        logger.info("Executing evaluations in parallel with thread-safe state management")
        tasks = [
            self._execute_turn_evaluations_with_progress(interaction_log),
            self._execute_node_evaluations_with_progress(interaction_log),
            self._execute_intent_evaluations_with_progress(interaction_log),
            self._execute_conversation_evaluations_with_progress(messages),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                evaluation_type = ["turn", "node", "intent", "conversation"][i]
                logger.error(f"Error in {evaluation_type} evaluation: {result}")

    async def _execute_turn_evaluations_with_progress(self, interaction_log: List[InteractionLog]) -> None:
        tracer = TracingManager().get_tracer()
        if tracer is not None:
            with tracer.start_as_current_span("eval/turn"):  # type: ignore
                try:
                    await self._execute_turn_evaluations(interaction_log)
                    await self.progress_handler.mark_milestone_complete(MilestoneType.TURN_EVALUATIONS)
                except Exception as e:
                    logger.error(f"Turn evaluations failed: {e}")
                    raise
        else:
            try:
                await self._execute_turn_evaluations(interaction_log)
                await self.progress_handler.mark_milestone_complete(MilestoneType.TURN_EVALUATIONS)
            except Exception as e:
                logger.error(f"Turn evaluations failed: {e}")
                raise

    async def _execute_node_evaluations_with_progress(self, interaction_log: List[InteractionLog]) -> None:
        tracer = TracingManager().get_tracer()
        if tracer is not None:
            with tracer.start_as_current_span("eval/node"):  # type: ignore
                try:
                    await self._execute_node_evaluations(interaction_log)
                    await self.progress_handler.mark_milestone_complete(MilestoneType.NODE_EVALUATIONS)
                except Exception as e:
                    logger.error(f"Node evaluations failed: {e}")
                    raise
        else:
            try:
                await self._execute_node_evaluations(interaction_log)
                await self.progress_handler.mark_milestone_complete(MilestoneType.NODE_EVALUATIONS)
            except Exception as e:
                logger.error(f"Node evaluations failed: {e}")
                raise

    async def _execute_intent_evaluations_with_progress(self, interaction_log: List[InteractionLog]) -> None:
        tracer = TracingManager().get_tracer()
        if tracer is not None:
            with tracer.start_as_current_span("eval/intent"):  # type: ignore
                try:
                    await self._execute_intent_evaluations(interaction_log)
                    await self.progress_handler.mark_milestone_complete(MilestoneType.INTENT_EVALUATIONS)
                except Exception as e:
                    logger.error(f"Intent evaluations failed: {e}")
                    raise
        else:
            try:
                await self._execute_intent_evaluations(interaction_log)
                await self.progress_handler.mark_milestone_complete(MilestoneType.INTENT_EVALUATIONS)
            except Exception as e:
                logger.error(f"Intent evaluations failed: {e}")
                raise

    async def _execute_conversation_evaluations_with_progress(self, messages: List[Message]) -> None:
        tracer = TracingManager().get_tracer()
        if tracer is not None:
            with tracer.start_as_current_span("eval/flow"):  # type: ignore
                try:
                    await self._execute_conversation_evaluations(messages)
                    await self.progress_handler.mark_milestone_complete(MilestoneType.CONVERSATION_EVALUATIONS)
                except Exception as e:
                    logger.error(f"Conversation evaluations failed: {e}")
                    raise
        else:
            try:
                await self._execute_conversation_evaluations(messages)
                await self.progress_handler.mark_milestone_complete(MilestoneType.CONVERSATION_EVALUATIONS)
            except Exception as e:
                logger.error(f"Conversation evaluations failed: {e}")
                raise

    async def _execute_turn_evaluations(self, interaction_log: List[InteractionLog]) -> None:
        message_history: List[Message] = []
        turn_evaluations: List[GranularEvaluationResults] = []
        for interaction in interaction_log:
            if interaction.llm_response.error or not interaction.llm_response.raw_content or interaction.llm_response.node not in self.nodes_under_evaluation:
                continue
            node = self.nodes_under_evaluation.get(interaction.llm_response.node)
            if not node or not node.evaluation_rules:
                continue
            turn_result = await TurnEvaluationStrategy(self.auto_refine_model_params).evaluate(
                self.context,
                {"node": node, "interaction": interaction, "message_history": message_history},
                self.turn_eval_model_params,
            )
            if turn_result.rule_results:
                for failed_rule in turn_result.failed_rules:
                    if failed_rule.node_name:
                        await self.failed_rules.add_failed_turn_rule(failed_rule.node_name, failed_rule)
                turn_evaluations.append(
                    GranularEvaluationResults(
                        category=Category.Turn,
                        input_text=interaction.llm_response.raw_content,
                        node=interaction.llm_response.node,
                        rule_results=turn_result.rule_results,
                    )
                )
            message_history.extend([
                HumanMessage(content=[{"type": "text", "text": interaction.user_message}]),
                AIMessage(content=interaction.llm_response.raw_content),
            ])
        await self.eval_state_manager.update_evaluations("turn_level_evaluations", turn_evaluations)

    async def _identify_node_blocks(self, interaction_log_list: List[InteractionLog]) -> List[NodeBlock]:
        if not interaction_log_list or len(interaction_log_list) == 1:
            return []
        node_block_list: List[NodeBlock] = []
        current_block: List[MiniInteractionLog] = []
        current_node_name: Optional[str] = None
        for interaction in interaction_log_list:
            if interaction.llm_response.error or not interaction.llm_response.raw_content or not interaction.llm_response.node or interaction.llm_response.node not in self.nodes_under_evaluation:
                continue
            node_name = interaction.llm_response.node
            mini_interaction = MiniInteractionLog(user_message=interaction.user_message, llm_response=interaction.llm_response.raw_content, node=node_name)
            if current_node_name is None:
                current_node_name = node_name
                current_block = [mini_interaction]
            elif node_name == current_node_name:
                current_block.append(mini_interaction)
            else:
                if len(current_block) > 1:
                    node_block_list.append(NodeBlock(node_name=current_node_name, mini_interaction_logs=current_block.copy()))
                current_node_name = node_name
                current_block = [mini_interaction]
        if len(current_block) > 1 and current_node_name is not None:
            node_block_list.append(NodeBlock(node_name=current_node_name, mini_interaction_logs=current_block))
        return node_block_list

    async def _execute_node_evaluations(self, interaction_log: List[InteractionLog]) -> None:
        node_blocks = await self._identify_node_blocks(interaction_log)
        if not node_blocks:
            logger.info("No node blocks found for evaluation")
            return
        node_evaluations: List[GranularEvaluationResults] = []
        for node_block in node_blocks:
            node_name = node_block.node_name
            node = self.nodes_under_evaluation.get(node_name)
            if not node or not node.evaluation_rules:
                continue
            
            tracer = TracingManager().get_tracer()
            if tracer is not None:
                with tracer.start_as_current_span(f"eval/node/{node_name}"): 
                    node_result = await NodeEvaluationStrategy(self.auto_refine_model_params).evaluate(
                        self.context, {"node": node, "interaction_history": node_block.mini_interaction_logs}, self.node_eval_model_params
                    )
            else:
                node_result = await NodeEvaluationStrategy(self.auto_refine_model_params).evaluate(
                    self.context, {"node": node, "interaction_history": node_block.mini_interaction_logs}, self.node_eval_model_params
                )
            if node_result.rule_results:
                for failed_rule in node_result.failed_rules:
                    if failed_rule.node_name:
                        await self.failed_rules.add_failed_node_rule(failed_rule.node_name, failed_rule)
                node_evaluations.append(
                    GranularEvaluationResults(
                        category=Category.Node,
                        input_text=Trinity.get_interaction_str(node_block.mini_interaction_logs),
                        node=node_name,
                        rule_results=node_result.rule_results,
                    )
                )
        await self.eval_state_manager.update_evaluations("node_level_evaluations", node_evaluations)

    async def _execute_intent_evaluations(self, interaction_log: List[InteractionLog]) -> None:
        state = await StateManager().get_state(self.context.tenant_id, self.context.agent_id, self.context.run_id)
        valid_intents = state.get_flow_data_valid_intents() if state and hasattr(state, 'get_flow_data_valid_intents') else {}
        message_history: List[Message] = []
        intent_evaluations: List[GranularEvaluationResults] = []
        for interaction in interaction_log:
            if interaction.llm_response.error or not interaction.llm_response.raw_content or not interaction.llm_response.intent_detector or interaction.llm_response.intent_detector not in self.nodes_under_evaluation:
                continue
            intent_result = await IntentEvaluationStrategy(self.auto_refine_model_params).evaluate(
                self.context,
                {"interaction": interaction, "conversation_str": Trinity.get_conversation_str(message_history), "valid_intents": valid_intents},
                self.intent_eval_model_params,
            )
            if intent_result.rule_results:
                for failed_rule in intent_result.failed_rules:
                    if failed_rule.node_name:
                        await self.failed_rules.add_failed_intent_rule(failed_rule.node_name, failed_rule)
                intent_evaluations.append(
                    GranularEvaluationResults(
                        category=Category.Intent,
                        input_text=f"Intent: {interaction.llm_response.intent}, Node: {interaction.llm_response.intent_detector}",
                        node=interaction.llm_response.intent_detector,
                        rule_results=intent_result.rule_results,
                    )
                )
            message_history.extend([
                HumanMessage(content=[{"type": "text", "text": interaction.user_message}]),
                AIMessage(content=interaction.llm_response.raw_content),
            ])
        await self.eval_state_manager.update_evaluations("intent_level_evaluations", intent_evaluations)

    async def _execute_conversation_evaluations(self, messages: List[Message]) -> None:
        state = await StateManager().get_state(self.context.tenant_id, self.context.agent_id, self.context.run_id)
        conversation_eval_rules = state.get_global_evaluation_rules() if state and hasattr(state, 'get_global_evaluation_rules') else []
        conversation_result = await ConversationEvaluationStrategy(self.auto_refine_model_params).evaluate(
            self.context,
            {"messages": messages, "conversation_eval_rules": conversation_eval_rules},
            self.conversation_eval_model_params,
        )
        if conversation_result.rule_results:
            for failed_rule in conversation_result.failed_rules:
                await self.failed_rules.add_failed_conversation_rule(failed_rule)
            input_text = Trinity.get_conversation_str(messages)
            conversation_evaluation = GranularEvaluationResults(category=Category.Flow, input_text=input_text, node="conversation", rule_results=conversation_result.rule_results)
            await self.eval_state_manager.update_evaluations("flow_level_evaluations", [conversation_evaluation])

    async def _update_final_state(self) -> None:
        await self.eval_state_manager.update_final_state(evaluated_nodes=list(self.nodes_under_evaluation.keys()))
        if self.request.payload and self.request.payload.persist_path:
            await self._save_evaluation_to_s3(persist_path=self.request.payload.persist_path)

    async def _save_evaluation_to_s3(self, persist_path: str) -> None:
        try:
            eval_state = await StateManager().get_eval_state(self.context.tenant_id, self.context.agent_id, self.context.run_id)
            if not eval_state or not eval_state.evaluation_result:
                logger.warning(f"No evaluation result found for {self.context.tenant_id}/{self.context.agent_id}/{self.context.run_id}")
                return
            bucket_name = persist_path.split("/")[0]
            object_key = "/".join(persist_path.split("/")[1:])
            json_data = json.dumps(eval_state.evaluation_result.model_dump(), indent=2, default=str)  # type: ignore
            try:
                Trinity.write_to_s3(bucket_name, object_key, json_data)
            except Exception:
                logger.error("S3 write not available in this environment")
        except Exception as e:
            logger.error(f"Error saving evaluation result to S3: {e}")

    async def _update_error_state(self) -> None:
        await self.eval_state_manager.update_error_state(evaluated_nodes=list(self.nodes_under_evaluation.keys()))

