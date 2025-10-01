## Phoenix Datasets Guide

This guide covers managing datasets with the Phoenix Python client, including how to create, read, list, append examples, and delete datasets.

### Prerequisites

- **Install**: `pip install arize-phoenix>=11.24.1`
- **Server URL**: Your Phoenix server base URL `https://dev-deusauditron.salescode.ai`
- **Authentication**: Create your PHOENIX_API_KEY from `https://dev-deusauditron.salescode.ai/profile`

### Initialize Client

```python
from phoenix.client import Client

client = Client(
    base_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
    api_key=os.getenv("PHOENIX_API_KEY"),
)
```

### Create a Dataset

Example according to the required parameters and their format in Scenario Evaluation endpoint

```python
from phoenix.client import Client
import os
import json

client = Client(
    base_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
    api_key=os.getenv("PHOENIX_API_KEY"),
)

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
      "content": "नमस्ते! Sir मैं वंशिका, कोका-कोला की सेल्स रिप्रेजेंटेटिव बोल रही हूँ। uh- क्या यह Mahavir Kirana Store का नंबर है?",
      "role": "assistant",
      "metadata": {
        "timestamp": "1755414404.322951",
        "node": "introduction"
      }
    },
  ]
}

expected_output = "This is the expected output"

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
                },
                {
                    "product": "coke 180 ml",
                    "mrp": 25.0,
                    "price per case": 720.22,
                    "price per piece": 20.01,
                    "case to piece quantity": 36.0,
                    "schemes": [],
                    "stock": ""
                },
            ]
        }
    },
    "post_call_catalogue": {
        "file": "Post-Call-Catalogue.csv",
        "data": [
            {
                "product": "dasani water 500 ml",
                "skuCode": "000000000000100632-ENRICH",
                "supplier": "DSB0039-ENRICH",
                "caseToPieceQuantity": 24.0
            },
            {
                "product": "thums up 750 ml",
                "skuCode": "000000000000103074-ENRICH",
                "supplier": "DSB0039-ENRICH",
                "caseToPieceQuantity": 24.0
            },
        ]
    }
}

dataset = client.datasets.create_dataset(
    name="docs-dataset",
    inputs=[{"Input": json.dumps(transcript, indent=2, ensure_ascii=False)}],
    outputs=[{"Output": expected_output}],
    metadata=[{"Meta Data": json.dumps(metadata, indent=2, ensure_ascii=False)}],
)

print(dataset.id, dataset.name)
```

### Read/Get a Dataset

```python
dataset = client.datasets.get_dataset(dataset="dataset-name")
print(dataset.id, dataset.name, dataset.examples)
```

### List Datasets

```python
datasets = client.datasets.list()
print(datasets)
```
