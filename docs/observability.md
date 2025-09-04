# Observability & Tracing

## Tracing

- Enable with `TRACING_ENABLED=true`
- Set `PHOENIX_COLLECTOR_ENDPOINT` to your collector (e.g., http://localhost:6006)
- All evaluation spans and LLM calls are traced where supported by the adapter

## Logging

- Context-aware logging with tenant/agent/run via `set_logging_context`
- Use INFO for operational logs, DEBUG for deep-dive
