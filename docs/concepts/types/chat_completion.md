# Chat Completion Batch

`ChatCompletionBatch` is the abstract batch class for processing chat completion requests. It's designed to utilize various Language Models (LLMs), using the OpenAI Chat Completion API format as its standard request structure.

1. **Universal Compatibility**: While based on the OpenAI format, `ChatCompletionBatch` can be used with LLMs from various providers, not just OpenAI.

2. **Extensibility**: Serves as a base class for creating specialized chat completion batch classes.

3. **Standardized Input/Output**: Uses OpenAI Batch File format and the OpenAI Chat Completion API format for input. Results are also consistently provided in the OpenAI format.

By standardizing on the OpenAI format, `ChatCompletionBatch` ensures consistency and interoperability across different LLM implementations within the LangBatch ecosystem.

## Initialize a Chat Completion Batch

You can initialize a ChatCompletionBatch by passing the path to a JSONL file. File should be in OpenAI batch File format and requests should be in OpenAI Chat Completion format.

```python
from langbatch import OpenAIChatCompletionBatch

batch = OpenAIChatCompletionBatch("data.jsonl")
```

You can also pass a list of requests to the batch.

```python
messages_list = [
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of the moon?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of the moon?"}
    ]}
]


batch = OpenAIChatCompletionBatch.create(messages_list)

# Initializing with request kwargs
batch = OpenAIChatCompletionBatch.create(
    messages_list, 
    request_kwargs={"temperature": 0.3, "max_tokens": 500}
)
```

## Get Results

```python
successful_results, unsuccessful_results = batch.get_results()
for result in successful_results:
    print(f"Content: {result['choices'][0]['message']['content']}")
```
