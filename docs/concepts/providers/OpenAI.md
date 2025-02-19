# OpenAI

## Data Format

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

## Create Chat Completion Batch

```python
import os
from langbatch import chat_completion_batch
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Create a batch object
batch = chat_completion_batch("path/to/batch-file.jsonl", provider="openai")

batch.start()
```

You can pass the OpenAI client to the `chat_completion_batch` function. This also allows you to use other providers with OpenAI Batch API compatible API. Model name should be changed to the provider's model name in requests in the batch file.

```python
from langbatch import chat_completion_batch
from openai import OpenAI
client = OpenAI(
    api_key='your-api-key',
    base_url="https://your-custom-endpoint.com/"
)

batch = chat_completion_batch(
    "path/to/batch-file.jsonl", 
    provider="openai", 
    client=client
)
```

## Create Embedding Batch

```json title="embedding-batch-file.jsonl"
{"custom_id": "6d292082-5c65-4b70-9900-a1418e49d5e7", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "I did not have a good day yesterday"}}
{"custom_id": "b8bf1ed2-2ec3-45f9-81b0-97813ad70ae4", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "I am feeling good today"}}
{"custom_id": "3cb5b179-464f-4ebe-b405-531caba76dd7", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Ill be having fun tomorrow"}}
```

```python
from langbatch import embedding_batch

batch = embedding_batch("path/to/embedding-batch-file.jsonl", provider="openai")
```

---

???+ tip
    Refer to [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch){:target="_blank"} for more information.