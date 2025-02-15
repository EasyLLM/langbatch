# Quickstart

## Prepare the batch file

```json title="batch-file.jsonl"
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

## Create a batch object
```python
from langbatch import chat_completion_batch

batch = chat_completion_batch("path/to/batch-file.jsonl", provider="openai")
```

## Start the batch job

Creating a batch object will not start the batch job. You need to start the batch job explicitly.

```python
batch.start()
```

## Check the status of the batch job
To check the status of the batch job, use the `get_status` method:

```python
status = batch.get_status()
print(status)
```

## Get the results of the batch job
After the batch job is successful, you can get the results using the `get_results` method:

```python
if batch.get_status() == "completed":
    successful_results, unsuccessful_results = batch.get_results()
    for result in successful_results:
        print(f"Custom ID: {result['custom_id']}")
        print(f"Content: {result['choices'][0]['message']['content']}")
```

???+ tip
    Learn more about the batch actions in the [Batch](../concepts/types/batch.md) page.
