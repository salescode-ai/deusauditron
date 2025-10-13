# End-to-End Scenario Evaluation Flow

This document describes the complete flow for ingesting data into Phoenix datasets and triggering evaluation using the scenario evaluation endpoint.

## Overview

The end-to-end flow consists of two main phases:

1. **Data Ingestion**: Creating and populating Phoenix datasets with conversation data
2. **Scenario Evaluation**: Running experiments over the datasets using the scenario evaluation endpoint

## Phase 1: Data Ingestion into Phoenix Datasets

### Prerequisites

- **Phoenix Server**: `https://dev-deusauditron.salescode.ai`
- **Authentication**: PHOENIX_API_KEY from the Phoenix profile page
- **Python Client**: `pip install arize-phoenix>=11.24.1`

### Step 1: Initialize Phoenix Client

```python
from phoenix.client import Client
import os

client = Client(
    base_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
    api_key=os.getenv("PHOENIX_API_KEY"),
)
```

### Step 2: Prepare Dataset Structure

Each dataset example requires three components:

#### Input: Conversation Transcript

```python
transcript = {
    "messages": [
        {
            "content": "हेलो sir!",
            "role": "assistant",
            "metadata": {
                "timestamp": "1755414399.422594"
            }
        },
        {
            "content": "हेलो?",
            "role": "user",
            "metadata": {
                "timestamp": "1755414404.322951"
            }
        },
        {
            "content": "नमस्ते! Sir मैं वंशिका, कोका-कोला की सेल्स रिप्रेजेंटेटिव बोल रही हूँ।",
            "role": "assistant",
            "metadata": {
                "timestamp": "1755414404.322951",
                "node": "introduction"
            }
        }
    ]
}
```

#### Output: Expected Result

```python
expected_output = "This is the expected output for this scenario"
```

#### Metadata: Catalog Data

```python
metadata = {
    "catalogs": {
        "precall_catalogue": {
            "file": "Precall-Catalogue-1.csv",
            "data": [
                {
                    "product": "coke 1 liter",
                    "mrp": 50.0,
                    "price per case": 660.03,
                    "price per piece": 44.0,
                    "case to piece quantity": 15.0,
                    "schemes": [
                        "Buy 1 case of coke 1 liter and get 1 bottle of coke 750 ml"
                    ],
                    "stock": ""
                }
            ]
        },
        "post_call_catalogue": {
            "file": "Post-Call-Catalogue.csv",
            "data": [
                {
                    "product": "dasani water 500 ml",
                    "skuCode": "000000000000100632-ENRICH",
                    "supplier": "DSB0039-ENRICH",
                    "caseToPieceQuantity": 24.0
                }
            ]
        }
    }
}
```

### Step 3: Create Dataset

```python
import json

dataset = client.datasets.create_dataset(
    name="my-scenario-dataset",
    inputs=[{"Input": json.dumps(transcript, indent=2, ensure_ascii=False)}],
    outputs=[{"Output": expected_output}],
    metadata=[{"Meta Data": json.dumps(metadata, indent=2, ensure_ascii=False)}],
)

print(f"Created dataset: {dataset.id} - {dataset.name}")
```

### Step 4: Add Additional Examples

```python
# Add more examples to existing dataset
client.datasets.add_examples_to_dataset(
    dataset="my-scenario-dataset",
    inputs=[
        {"Input": json.dumps(transcript, indent=2, ensure_ascii=False)},
        {"Input": json.dumps(another_transcript, indent=2, ensure_ascii=False)}
    ],
    outputs=[
        {"Output": expected_output},
        {"Output": another_expected_output}
    ],
    metadata=[
        {"Meta Data": json.dumps(metadata, indent=2, ensure_ascii=False)},
        {"Meta Data": json.dumps(another_metadata, indent=2, ensure_ascii=False)}
    ],
)
```

## Phase 2: Scenario Evaluation Execution

### Step 1: Prepare Evaluation Payload

```python
payload = {
    "datasetName": "existing-dataset-name",
    "agentName": "lob-agentName",
    "experimentName": "testing1",
    "metadata": {
        "key1": "value1",
    },
    "blueprint": "any agent blueprint",
    "replay": False,  # Set to True for message-by-message replay
    # Optional (will be stored in Phoenix Experiment metadata)
    "outlet_id": "outlet-id",
    "outlet_name": "outlet-name"
}
```

### Step 2: Trigger Scenario Evaluation

Trigger from UI using the above payload and endpoint `/scenario/run`.

### Step 3: Response

```json
{
  "success": true,
  "experiment_id": ["abc"]
}
```

### Step 4: Evaluation Result on UI

Use the experiment_id from the response to fetch the Evaluation result from endpoint - `https://dev-deusauditron.salescode.ai/v1/experiments/{experiment_id}/json`

## How to show Dataset on UI

### Dataset Naming Convention

We are storing the datasets using the following convention: agentName = `{lob}-{agentName}`

### Fetch Datasets according to agentName

To get the datasets for a particular agent, use the following endpoint: `https://dev-deusauditron.salescode.ai/v1/datasets?name={agentName}`

### Get all Experiments of a Dataset

`https://dev-deusauditron.salescode.ai/v1/datasets/{dataset_id}/experiments`
