# Architecture Overview

Deusauditron is a standalone FastAPI service focused on agent evaluation.

## High-level Components

- `app.py`: FastAPI app factory, CORS, engine bootstrap, routers registration.
- `routers/evaluation.py`: Evaluation endpoints (POST/GET/DELETE).
- `engine.py`: Auditron engine handling eval queueing, background consumers, and request handling.
- `eval/`: Core evaluation pipeline
  - `eval_worker.py`: `LLMEvaluator` orchestrating evaluation lifecycle
  - `eval_strategies.py`: Turn/Node/Intent/Conversation strategies
  - `refinement_strategies.py`: Auto-refinement orchestration
  - `eval_utils.py`, `eval_common.py`: helpers, model params, parsing
  - `progress_handler.py`, `progress_tracker.py`: progress tracking
  - `eval_state_manager.py`: atomic eval state updates
- `state/`: StateManager + stores (local/redis) for eval state and runtime artifacts
- `queue/`: Queue backends (local/redis) for async request handling
- `lock/`: Lock managers (local/redis) for concurrency control
- `llm_abstraction/`: LLMInvoker + adapters for provider-agnostic calls
- `logging/`: context-aware logging wrappers
- `config/`: environment-driven configuration and tracing manager
- `schemas/`: Pydantic (autogen + shared models)
- `prompts/`: Prompt templates used by strategies

## Runtime Flow

- FastAPI app starts Auditron engine (background consumer tasks start)
- Requests hit evaluation router → engine enqueues → background worker consumes
- `LLMEvaluator` loads state, runs strategies in parallel (async gather), updates atomic eval state, persists if configured
