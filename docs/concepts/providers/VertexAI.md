# Vertex AI

You can utilize Vertex AI Batch generations for running batch generations on Vertex AI models via LangBatch.

## Data Format

OpenAI data format can be used in LangBatch for Vertex AI. But the model name can be skipped here. And all the requests should have same model name.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

## Vertex AI Initialization

Vertex AI should be initialized with the project id and location.

```python
import os
import vertexai

GCP_PROJECT = os.environ.get('GCP_PROJECT')
GCP_LOCATION = os.environ.get('GCP_LOCATION')

vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
```

## Vertex AI Batch

Vertex AI Batch can be created with the model name, project id, location, bigquery input dataset and bigquery output dataset values.

```python
from langbatch.vertexai import VertexAIChatCompletionBatch

batch = VertexAIChatCompletionBatch(
    file="data.jsonl",
    model_name="gemini-1.5-flash-002",
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    bigquery_input_dataset="batches",
    bigquery_output_dataset="gen_ai_batch_prediction")

batch.start()
```

!!!note
    You need to make sure that the BigQuery datasets are created before running the batch.