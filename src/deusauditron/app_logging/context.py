import contextvars
from typing import Optional

_tenant_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("tenant_id", default=None)
_agent_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("agent_id", default=None)
_run_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("run_id", default=None)


def set_logging_context(tenant_id: str, agent_id: str, run_id: str) -> None:
    _tenant_id_var.set(tenant_id)
    _agent_id_var.set(agent_id)
    _run_id_var.set(run_id)


def get_context_prefix() -> str:
    tenant_id = _tenant_id_var.get()
    agent_id = _agent_id_var.get()
    run_id = _run_id_var.get()
    if all([tenant_id, agent_id, run_id]):
        return f"{tenant_id}/{agent_id}/{run_id}"
    return ""


def clear_logging_context() -> None:
    _tenant_id_var.set(None)
    _agent_id_var.set(None)
    _run_id_var.set(None)

