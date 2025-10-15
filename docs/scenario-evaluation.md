## Scenario Evaluation Router

This document explains the end-to-end flow of the scenario evaluation API defined in `src/deusauditron/routers/scenario_evaluation.py`, including Phoenix client setup, dataset retrieval, experiment execution, custom task/evaluator functions, environment variables, and troubleshooting.

### Overview

- **Route**: `POST /scenario/run`
- **Purpose**: Run a Phoenix experiment over a Phoenix dataset where each example triggers a scenario execution in the DeusMachine agent, then evaluate the outputs with a custom evaluator.
- **Core libs**: `fastapi`, `phoenix` (client and experiments), `nest_asyncio`

### Phoenix Client Initialization

The router initializes a Phoenix client once at import time:

```python
import os
import phoenix as px

client = px.Client(
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
    api_key=os.getenv("PHOENIX_API_KEY")
)
```

### Request Payload

`ScenarioPayload` (from `deusauditron.schemas.shared_models.models`) includes at least:

- `dataset_name`: Phoenix dataset name to load
- `agent_name`, `experiment_name`: for tagging experiment metadata
- `metadata`, `blueprint`, `dynamic_data`: Variables for DM
- `transcript`: conversation messages for the agent
- `replay`: If True, then try to simulate each and every message otherwise just pass in the messages field of DM create agent request.
- `outlet_id`: Optional variable to be stored in Experiment metadata
- `outlet_name`: Optional variable to be stored in Experiment metadata

### Dataset Retrieval

The router fetches the dataset by name from Phoenix:

```python
dataset = client.get_dataset(name=payload.dataset_name)
```

### Experiment Execution

The route calls `run_experiment` from `phoenix.experiments`:

```python
from phoenix.experiments import run_experiment

experiment = run_experiment(
    dataset=dataset,
    task=scenario_task,
    evaluators=[scenario_evaluator],
    experiment_metadata={
        "agent_name": payload.agent_name,
        "experiment_name": payload.experiment_name,
    },
)
```

Phoenix will iterate over dataset examples, invoke `scenario_task` for each, and then call `scenario_evaluator` with the output and expected values.

### The Task Function (`scenario_task`)

Purpose: Convert the example into inputs for DeusMachine and produce a final string output.

Key steps:

1. Parse example input JSON (expects `messages` array) and build a transcript (list of `Message`).
2. Parse example metadata JSON (expects `catalogs`) and construct `dynamic_data`.
3. Call `run_task(...)` to execute the DeusMachine agent with transcript/dynamic data and return the final output.

### The Evaluator Function (`scenario_evaluator`)

Purpose: Compare the model output against the expected output and return an `EvaluationResult`.

Logic:

- If the task returned an error string, label the result as `ERROR` with score `0.0`.
- Otherwise, call `EvaluationUtils().custom_evaluator(final_output, expected_output, dm_adapter)` and map the result to `score`, `label`, and `explanation`.

Return type is `phoenix.experiments.types.EvaluationResult`.

### Running the Agent (`run_task`)

Purpose: Create a DeusMachine agent, run it with transcript input (either replayed or single-turn), and return the final output string.

Flow:

1. Generate `tenant_id`, `agent_id`, `run_id` and set logging context.
2. Create agent via `DMAdapter.create_agent(...)` with metadata/blueprint/dynamic_data.
3. Run agent either once (non-replay) or over user messages (replay).
4. Extract `final_output` from the agent result.
5. Ensure cleanup by deleting the agent in `finally`.

### End-to-End Flow Summary

1. Receive payload on `POST /scenario/run`.
2. Load Phoenix dataset by name.
3. Execute `run_experiment` with custom task and evaluator.
4. Return `{ success: true, experiment_id }` or 500 with error.
