from typing import Any, Optional

from deusauditron.app_logging.logger import logger
from deusauditron.config import TracingManager


def _safe_get(d: Any, *keys: str) -> Optional[Any]:
    try:
        for k in keys:
            if d is None:
                return None
            if isinstance(d, dict):
                if k in d:
                    return d.get(k)
            else:
                if hasattr(d, k):
                    return getattr(d, k)
        return None
    except Exception:
        return None


def _boolify(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "pass", "passed", "yes", "y"):
                return True
            if v in ("false", "fail", "failed", "no", "n"):
                return False
    except Exception:
        return None
    return None


def log_rule_evaluation_span(
    *,
    tenant_id: str,
    agent_id: str,
    run_id: str,
    category: str,
    node_name: Optional[str],
    rule_result: Any,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    try:
        tracer = TracingManager().get_tracer()
        if tracer is None:
            return

        rule_name = (
            _safe_get(rule_result, "rule_name")
            or _safe_get(rule_result, "name")
            or _safe_get(rule_result, "rule")
        )
        passed = (
            _boolify(_safe_get(rule_result, "passed"))
            or _boolify(_safe_get(rule_result, "is_passed"))
            or _boolify(_safe_get(rule_result, "pass_flag"))
        )
        score = _safe_get(rule_result, "score") or _safe_get(rule_result, "value")
        explanation = (
            _safe_get(rule_result, "explanation")
            or _safe_get(rule_result, "reason")
            or _safe_get(rule_result, "message")
        )

        span_name = f"eval/{category}/{rule_name or 'unknown'}"
        with tracer.start_as_current_span(span_name):  # type: ignore
            try:
                import opentelemetry.trace as otel_trace  # type: ignore
                span = otel_trace.get_current_span()
                span.set_attribute("app.tenant_id", tenant_id)
                span.set_attribute("app.agent_id", agent_id)
                span.set_attribute("app.run_id", run_id)
                span.set_attribute("app.eval.category", category)
                if source:
                    span.set_attribute("app.eval.source", source)
                if node_name:
                    span.set_attribute("app.eval.node", node_name)
                if rule_name:
                    span.set_attribute("app.eval.rule_name", str(rule_name))
                if passed is not None:
                    span.set_attribute("app.eval.passed", bool(passed))
                if score is not None:
                    try:
                        span.set_attribute("app.eval.score", float(score))
                    except Exception:
                        span.set_attribute("app.eval.score_raw", str(score))
                if explanation:
                    span.set_attribute("app.eval.explanation", str(explanation))
                if input_text:
                    span.set_attribute("app.eval.input", input_text[:4096])
                if output_text:
                    span.set_attribute("app.eval.output", output_text[:4096])
            except Exception:
                pass
    except Exception as e:
        try:
            logger.debug(f"Phoenix adapter log_rule_evaluation_span error: {e}")
        except Exception:
            pass


