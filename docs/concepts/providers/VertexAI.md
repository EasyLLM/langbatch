# Vertex AI

You can run batch inference jobs on Gemini, Claude and Llama models available in Vertex AI via LangBatch.

## Data Format

OpenAI data format can be used in LangBatch for Vertex AI. But the model name can be skipped here.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

???+note
    In VertexAI, you can only send requests to a single model in a batch. If you want to use multiple models, you need to create multiple batches.

## Vertex AI Initialization

Vertex AI should be initialized with the project id and location.

???+note
    You need to use the correct location according to the model you are using. Check this [link](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations){:target="_blank"} for more available locations.

```python
import os
import vertexai

GCP_PROJECT = os.environ.get('GCP_PROJECT')
GCP_LOCATION = os.environ.get('GCP_LOCATION')
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
```

!!! tip
    You can use a service account to avoid the frequent authentication error `google.auth.exceptions.RefreshError: Reauthentication is needed. Please run "gcloud auth application-default login" to reauthenticate`. A service account is long-lived as it does not have an expiry time. You can create a service account with only required permissions - `Vertex AI user`, `BigQuery Data Editor` and `BigQuery User`. Check [Service Account Creation](https://skypilot.readthedocs.io/en/latest/cloud-setup/cloud-permissions/gcp.html#service-account){:target="_blank"} for more information.

You can either set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file or use the following code to set the credentials.

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service-account-key.json"
```

## Create Chat Completion Batch

Vertex AI requires project and BigQuery datasets configuration. In Vertex AI, you can only use a single model for all requests in a batch. So you need to pass the model name to the `chat_completion_batch` function.

```python
import os
from langbatch import chat_completion_batch

os.environ["GCP_PROJECT"] = "your-gcp-project"
os.environ["GCP_BIGQUERY_INPUT_DATASET"] = "your-input-dataset"
os.environ["GCP_BIGQUERY_OUTPUT_DATASET"] = "your-output-dataset"

batch = chat_completion_batch(
    "path/to/batch-file.jsonl", 
    provider="vertex_ai", 
    model="gemini-2.0-flash-001"
)
```

You can also pass the configuration values as arguments:

```python
batch = chat_completion_batch(
    "path/to/batch-file.jsonl", 
    provider="vertex_ai",
    model="gemini-2.0-flash-001",
    gcp_project="your-gcp-project",
    bigquery_input_dataset="your-input-dataset",
    bigquery_output_dataset="your-output-dataset"
)
```

!!!info
    You need to make sure that the BigQuery datasets are created before running the batch. They need to be in the same project and location as the Vertex AI Batch.

## Partner Models

You can also use the partner models available in VertexAI. Claude from Anthropic and Llama from Meta are available in VertexAI. You need to enable them in below links before using them. 

- [Claude 3.5 Sonnet v2](https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden/claude-3-5-sonnet-v2){:target="_blank"}, [Claude 3.5 Haiku](https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden/claude-3-5-haiku){:target="_blank"}, [Llama 3.1 models](https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.1-405b-instruct-maas){:target="_blank"}, [Llama 3.3 70B](https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.3-70b-instruct-maas){:target="_blank"}

Available models:

- Claude 3.5 Sonnet v2 (claude-3-5-sonnet-v2@20241022)
- Claude 3.5 Haiku (claude-3-5-haiku@20241022)
- Llama 3.1 405B (llama-3.1-405b-instruct-maas)
- Llama 3.3 70B (llama-3.3-70b-instruct-maas)
- Llama 3.1 70B (llama-3.1-70b-instruct-maas)
- Llama 3.1 8B (llama-3.1-8b-instruct-maas)