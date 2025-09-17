import os
from typing import List, Optional

from deusauditron.app_logging.logger import logger
from deusauditron.observability.phoenix_rubric_adapter import maybe_run_rubric_for_rule
from deusauditron.observability.phoenix_eval_adapter import log_rule_evaluation_span


def _enabled() -> bool:
    try:
        return os.getenv("PHOENIX_OOB_EVALS", "false").lower() == "true"
    except Exception:
        return False


def _get_enabled_evaluators() -> List[str]:
    raw = os.getenv("PHOENIX_OOB_EVALUATORS", "helpfulness,hallucination").strip()
    items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    allowed = {"helpfulness", "hallucination"}
    return [s for s in items if s in allowed]


def _build_rubric_text(name: str) -> str:
    if name == "helpfulness":
        return (
            "Judge if the output directly and helpfully addresses the user's input. "
            "Pass if it is on-topic, actionable, and answers the question."
        )
    if name == "hallucination":
        return (
            "Judge if the output stays faithful to the provided input/context. "
            "Fail if the output contains facts not supported by the input/context."
        )
    return "Judge the output according to the rubric."  # fallback


async def maybe_run_oob_evals_for_turn(
    *,
    tenant_id: str,
    agent_id: str,
    run_id: str,
    node_name: Optional[str],
    user_message: Optional[str],
    assistant_output: Optional[str],
    conversation_context: Optional[str],
) -> None:
    if not _enabled():
        return
    evaluators = _get_enabled_evaluators()
    if not evaluators:
        return
    for ev in evaluators:
        try:
            rubric = _build_rubric_text(ev)
            rule_name = ev.capitalize()
            # Prefer Phoenix OOB evaluators via experiments if available else fallback to rubric
            try:
                from phoenix.experiments.evaluators import HelpfulnessEvaluator  # type: ignore
                from phoenix.experiments.evaluators import HallucinationEvaluator  # type: ignore
                from phoenix.evals.models import OpenAIModel  # type: ignore
                model = OpenAIModel()
                if ev == "helpfulness":
                    evaluator = HelpfulnessEvaluator(model=model)
                elif ev == "hallucination":
                    evaluator = HallucinationEvaluator(model=model)
                else:
                    evaluator = None
                if evaluator is not None and assistant_output:
                    score = None
                    try:
                        # Minimal call style: most evaluators accept (input, output) or (context, output)
                        # We call with output and optional input/context; the evaluator will compute a score/label
                        if user_message is not None:
                            score = evaluator(output=assistant_output, input=user_message)
                        elif conversation_context is not None:
                            score = evaluator(output=assistant_output, input=conversation_context)
                        else:
                            score = evaluator(output=assistant_output)
                    except Exception:
                        score = None
                    log_rule_evaluation_span(
                        tenant_id=tenant_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        category="out_of_box",
                        node_name=node_name,
                        rule_result={"rule_name": rule_name, "score": score},
                        input_text=user_message or conversation_context,
                        output_text=assistant_output,
                        source="phoenix_oob",
                    )
                    continue
            except Exception:
                pass

            await maybe_run_rubric_for_rule(
                tenant_id=tenant_id,
                agent_id=agent_id,
                run_id=run_id,
                category="out_of_box",
                node_name=node_name,
                rule_name=rule_name,
                rubric=rubric,
                input_text=user_message,
                output_text=assistant_output,
                context_text=conversation_context,
            )
        except Exception as e:
            logger.debug(f"OOB evaluator '{ev}' failed for node '{node_name}': {e}")


async def maybe_run_oob_evals_for_flow(
    *,
    tenant_id: str,
    agent_id: str,
    run_id: str,
    conversation_context: Optional[str],
) -> None:
    if not _enabled():
        return
    evaluators = _get_enabled_evaluators()
    if not evaluators:
        return
    for ev in evaluators:
        try:
            rubric = _build_rubric_text(ev)
            rule_name = ev.capitalize()
            try:
                from phoenix.experiments.evaluators import HelpfulnessEvaluator  # type: ignore
                from phoenix.experiments.evaluators import HallucinationEvaluator  # type: ignore
                from phoenix.evals.models import OpenAIModel  # type: ignore
                model = OpenAIModel()
                if ev == "helpfulness":
                    evaluator = HelpfulnessEvaluator(model=model)
                elif ev == "hallucination":
                    evaluator = HallucinationEvaluator(model=model)
                else:
                    evaluator = None
                if evaluator is not None and conversation_context:
                    score = None
                    try:
                        score = evaluator(output=conversation_context)
                    except Exception:
                        score = None
                    log_rule_evaluation_span(
                        tenant_id=tenant_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        category="out_of_box",
                        node_name="conversation",
                        rule_result={"rule_name": rule_name, "score": score},
                        input_text=None,
                        output_text=None,
                        source="phoenix_oob",
                    )
                    continue
            except Exception:
                pass

            await maybe_run_rubric_for_rule(
                tenant_id=tenant_id,
                agent_id=agent_id,
                run_id=run_id,
                category="out_of_box",
                node_name="conversation",
                rule_name=rule_name,
                rubric=rubric,
                input_text=None,
                output_text=None,
                context_text=conversation_context,
            )
        except Exception as e:
            logger.debug(f"OOB evaluator '{ev}' failed for flow: {e}")


