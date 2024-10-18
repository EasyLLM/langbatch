# OpenAI

You can utilize OpenAI Batch API for running batch generations on OpenAI models via LangBatch.

## Data Format

OpenAI batch data format can be used for OpenAI.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

## OpenAI Client

OpenAI client is used to make requests to the OpenAI service.

```python
from openai import OpenAI
from langbatch.openai import OpenAIChatCompletionBatch

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

batch = OpenAIChatCompletionBatch(
    file="data.jsonl",
    client=client
)

batch.start()
```

Refer to [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch){:target="_blank"} for more information.

## Other Providers with OpenAI API support

If you are using other providers with OpenAI Batch API compatible API, you can just change the base_url to the provider's base_url.
And change the model name too.

```python
from openai import OpenAI
from langbatch.openai import OpenAIChatCompletionBatch

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.provider.com/v1/"
)

batch = OpenAIChatCompletionBatch(
    file="data.jsonl",
    client=client
)
```