# Anthropic

You can run batch inference jobs on Anthropic models like Claude 3.5 Sonnet via LangBatch.

## Data Format

OpenAI batch data format can be used for Anthropic. But make sure to use the Claude model names in the batch file.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "claude-3-5-haiku-20241022", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "claude-3-5-haiku-20241022", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

## Create Chat Completion Batch

```python
import os
from langbatch import chat_completion_batch
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

batch = chat_completion_batch("path/to/batch-file.jsonl", provider="anthropic")
```

You can also pass the Anthropic client as an argument to the `chat_completion_batch` function.

```python
from langbatch import chat_completion_batch
from anthropic import Anthropic

client = Anthropic(api_key="your-anthropic-api-key")
batch = chat_completion_batch(
    "path/to/batch-file.jsonl", 
    provider="anthropic", 
    client=client
)
```

Refer to [Anthropic Batch API Documentation](https://docs.anthropic.com/en/docs/build-with-claude/message-batches){:target="_blank"} for more information.