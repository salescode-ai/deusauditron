# API Reference

Base path: `/api/v1`

## Start Evaluation

POST `/agents/{tenant_id}/{agent_id}/{run_id}/evaluation`

Body (example):

```json
{
  "api_keys": { "openai": "sk-..." },
  "node_names": ["welcome_node"],
  "auto_refine": true,
  "persist_path": "my-bucket/path/eval.json",
  "force": false
}
```

Responses:

- 200: `EvaluationResult` (status Requested)
- 404: agent state not found
- 409: evaluation already requested (unless `force=true` with Completed/Error)
- 503/500: service unavailable/internal error

## Get Evaluation

GET `/agents/{tenant_id}/{agent_id}/{run_id}/evaluation`

Query:

- `evaluation_path` (optional) – S3 path fallback

Responses:

- 200: `EvaluationResult` (current state)
- 404: never requested / not found

## Delete Evaluation

DELETE `/agents/{tenant_id}/{agent_id}/{run_id}/evaluation`

Responses:

- 200: deleted
- 404: never requested / not found
- 409: in progress

## Models

- `EvaluationPayload`: api_keys, node_names, auto_refine, persist_path, force
- `EvaluationResult`: status, progress, turn/node/flow/intent evaluations, auto_refinements, timestamps, evaluated_nodes
