import json
import os
import re
from typing import Any, Dict, List, Optional

from deusauditron.app_logging.logger import logger
from deusauditron.llm_abstraction.llm_helper import LLMInvoker
from deusauditron.schemas.shared_models.models import Message, MessageRole
from deusauditron.observability.phoenix_eval_adapter import log_rule_evaluation_span


def _enabled() -> bool:
    try:
        return os.getenv("PHOENIX_RUBRIC_EVALS", "true").lower() == "true"
    except Exception:
        return False


def _build_rubric_prompt(rule_name: str, rubric: str, input_text: Optional[str], output_text: Optional[str], context_text: Optional[str]) -> str:
    parts: List[str] = []
    parts.append("You are an expert evaluator. Grade the candidate output strictly using the rubric.")
    parts.append("Return a compact JSON object with keys: pass (boolean) and explanation (string).")
    parts.append("")
    parts.append(f"Rule: {rule_name}")
    parts.append(f"Rubric: {rubric}")
    if context_text:
        parts.append("Context:")
        parts.append(context_text)
    if input_text:
        parts.append("Input:")
        parts.append(input_text)
    if output_text:
        parts.append("Output:")
        parts.append(output_text)
    parts.append("")
    parts.append("Respond ONLY with JSON like: {\"pass\": true, \"explanation\": \"...\"}")
    return "\n".join(parts)


async def _judge_with_gpt4o(prompt: str) -> Dict[str, Any]:
    try:
        response = await LLMInvoker.get_instance().invoke_non_streaming(
            llm_request=[Message(content=prompt, role=MessageRole.USER)],
            model_name="groq/openai/gpt-oss-120b",
            temperature=0.0,
            reasoning=False,
        )
        if response.error or not response.content:
            return {"pass": None, "explanation": "No response from judge"}
        text = response.content.strip()
        try:
            return json.loads(text)
        except Exception:
            # Attempt to strip code fences
            if text.startswith("```") and text.endswith("```"):
                text = text.strip().strip("`")
            try:
                return json.loads(text)
            except Exception:
                # Heuristic fallback: detect a boolean for "pass" via regex
                lowered = text.lower()
                match = re.search(r'"pass"\s*:\s*(true|false)', lowered)
                if match:
                    passed_val = match.group(1) == "true"
                else:
                    passed_val = None
                return {"pass": passed_val, "explanation": text[:2000]}
    except Exception as e:
        logger.debug(f"Rubric judge error: {e}")
        return {"pass": None, "explanation": f"Judge error: {e}"}


async def maybe_run_rubric_for_rule(
    *,
    tenant_id: str,
    agent_id: str,
    run_id: str,
    category: str,
    node_name: Optional[str],
    rule_name: str,
    rubric: str,
    input_text: Optional[str],
    output_text: Optional[str],
    context_text: Optional[str],
) -> None:
    if not _enabled():
        return
    try:
        logger.debug(
            f"[PhoenixRubric] Enabled: judging rule='{rule_name}' category='{category}' node='{node_name or ''}'"
        )
        prompt = _build_rubric_prompt(rule_name, rubric, input_text, output_text, context_text)
        result = await _judge_with_gpt4o(prompt)
        passed = result.get("pass")
        explanation = result.get("explanation")
        logger.debug(
            f"[PhoenixRubric] Result rule='{rule_name}': pass={passed} explanation_len={len(explanation) if isinstance(explanation, str) else 'n/a'}"
        )
        log_rule_evaluation_span(
            tenant_id=tenant_id,
            agent_id=agent_id,
            run_id=run_id,
            category=category,
            node_name=node_name,
            rule_result={"rule_name": rule_name, "passed": passed, "explanation": explanation},
            input_text=input_text or context_text,
            output_text=output_text,
            source="custom_rubric",
        )
    except Exception as e:
        logger.debug(f"Phoenix rubric adapter failed for rule '{rule_name}': {e}")


