# Evaluation Call Flow

## 1. Request Entry (POST)

- Route: `POST /api/v1/agents/{tenant_id}/{agent_id}/{run_id}/evaluation`
- Body: `EvaluationPayload` (api_keys, node_names, auto_refine, persist_path, force)
- Steps:
  1. Set logging context
  2. Load agent state and existing eval state
  3. Conflict handling (409) unless `force=true` and previous is Completed/Error
  4. Create new `EvalState` with `EvaluationResult(status=Requested, progress=0)`
  5. Enqueue `AgentEvalRequest` via queue backend

## 2. Background Processing

- Engine consumer loop dequeues `AgentEvalRequest`
- Spawns task: `LLMEvaluator.evaluate()`

## 3. LLMEvaluator Lifecycle

1. `_load_data_and_prepare()`
   - Load state, eval_state
   - Merge API keys (state + request payload)
   - Identify nodes under evaluation
   - Build model params (turn/node/intent/flow/auto-refine)
   - Mark start state (In_Progress, progress=0)
2. `_execute_evaluations_parallel()`
   - Gather turn/node/intent/flow in parallel (async)
   - Each evaluation uses strategies and shared utils; failures recorded thread-safely
3. Optional `_auto_refine_evaluations()`
   - Orchestrator analyzes failed rules and proposes refinements
4. `_update_final_state()`
   - Status=Completed, progress=100, evaluated_nodes set
   - Optional S3 persistence if `persist_path` provided

## 4. Status (GET)

- Route: `GET /api/v1/agents/{tenant_id}/{agent_id}/{run_id}/evaluation`
- Looks up eval result from store; optional S3 fallback

## 5. Deletion (DELETE)

- Route: `DELETE /api/v1/agents/{tenant_id}/{agent_id}/{run_id}/evaluation`
- Deletes eval state if not In_Progress
