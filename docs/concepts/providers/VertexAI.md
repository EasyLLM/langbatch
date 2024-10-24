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

!!! tip
    You can use a service account to avoid the frequent authentication error `google.auth.exceptions.RefreshError: Reauthentication is needed. Please run "gcloud auth application-default login" to reauthenticate`. A service account is long-lived as it does not have an expiry time. You can create a service account with only required permissions - `Vertex AI user`, `BigQuery Data Editor` and `BigQuery User`. Check [Service Account Creation](https://skypilot.readthedocs.io/en/latest/cloud-setup/cloud-permissions/gcp.html#service-account) for more information.

You can either set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file or use the following code to set the credentials.

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service-account-key.json"
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

!!!info
    You need to make sure that the BigQuery datasets are created before running the batch.