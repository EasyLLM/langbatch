# Vertex AI

You can utilize Vertex AI Batch generations for running batch generations on Vertex AI models via LangBatch.

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
    You need to use the correct location according to the model you are using. Check this [link](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations) for more available locations.

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
    You need to make sure that the BigQuery datasets are created before running the batch. They need to be in the same project and location as the Vertex AI Batch.

## Partner Models

You can also use the partner models available in VertexAI. Claude from Anthropic and Llama from Meta are available in VertexAI. You need to enable them in below links before using them. 

- [Claude 3.5 Sonnet v2](https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden/claude-3-5-sonnet-v2)
- [Claude 3.5 Haiku v2](https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden/claude-3-5-haiku)
- [Llama 3.1 models](https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.1-405b-instruct-maas)

### VertexAI Claude Batch

Available models:

- Claude 3.5 Sonnet v2 (claude-3-5-sonnet-v2@20241022)
- Claude 3.5 Haiku (claude-3-5-haiku@20241022)

```python
from langbatch.vertexai import VertexAIClaudeChatCompletionBatch

batch = VertexAIClaudeChatCompletionBatch(
    file="data.jsonl",
    model_name="claude-3-5-sonnet-v2@20241022",
    project=GCP_PROJECT,
    location='us-east5',
    bigquery_input_dataset="batches",
    bigquery_output_dataset="gen_ai_batch_prediction")

batch.start()
```

### VertexAI Llama Batch

Available models:

- Llama 3.1 405B (llama-3.1-405b-instruct-maas)
- Llama 3.1 70B (llama-3.1-70b-instruct-maas)
- Llama 3.1 8B (llama-3.1-8b-instruct-maas)

```python
from langbatch.vertexai import VertexAILlamaChatCompletionBatch

batch = VertexAILlamaChatCompletionBatch(
    file="data.jsonl",
    model_name="llama-3.1-405b-instruct-maas",
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    bigquery_input_dataset="batches",
    bigquery_output_dataset="gen_ai_batch_prediction")

batch.start()
```