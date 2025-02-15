# Chat Completion Batch

`ChatCompletionBatch` is the abstract batch class for processing chat completion requests. It's designed to utilize various Language Models (LLMs), using the OpenAI Chat Completion API format for requests and responses.

By standardizing on the OpenAI format, `ChatCompletionBatch` ensures consistency and interoperability across different LLM providers within the LangBatch.

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

# Initializing with request kwargs and batch kwargs
from openai import OpenAI
client = OpenAI(api_key="your-api-key", base_url="provider-base-url")
batch = OpenAIChatCompletionBatch.create(
    messages_list, 
    request_kwargs={"temperature": 0.3, "max_tokens": 500},
    batch_kwargs={ "client": client }
)
```

???+ info
    You can only pass the 'messages' list in the requests here. And the provided `request_kwargs` will be applied to all requests in the batch. This is useful for cases where you want to use the same inference configuration to all requests in the batch and only the 'messages' list is different for each request.

## Get Results

In ChatCompletionBatch, the successful results contain `choices` and `custom_id` keys.

```python
successful_results, unsuccessful_results = batch.get_results()
for result in successful_results:
    print(f"Custom ID: {result['custom_id']}")
    print(f"Content: {result['choices'][0]['message']['content']}")
```
