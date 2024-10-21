# Anthropic

You can utilize Anthropic Batch API for running batch generations on Anthropic models like Claude 3.5 Sonnet via LangBatch.

## Data Format

OpenAI batch data format can be used for Anthropic.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

## Anthropic Client

Anthropic client is used to make requests to the Anthropic service.

```python
from anthropic import Anthropic
from langbatch.anthropic import AnthropicChatCompletionBatch

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

batch = AnthropicChatCompletionBatch(
    file="data.jsonl",
    client=client
)

batch.start()
```

Refer to [Anthropic Batch API Documentation](https://docs.anthropic.com/en/docs/build-with-claude/message-batches){:target="_blank"} for more information.