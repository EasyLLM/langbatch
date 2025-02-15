# Running a Batch with LangBatch

LangBatch provides a simple interface to run batch jobs.

```python
from langbatch import OpenAIChatCompletionBatch

# Create a batch object
batch = OpenAIChatCompletionBatch("path/to/file.jsonl")

# Start the batch job
batch.start()
```

To check the status of the batch job, use the `get_status` method:

```python
status = batch.get_status()
print(status)
```

To get the results of the batch job, use the `get_results` method:

```python
successful_results, unsuccessful_results = batch.get_results()
for result in successful_results:
    print(f"Custom ID: {result['custom_id']}")
    print(f"Content: {result['choices'][0]['message']['content']}")
```

!!! tip
    You can perform the same actions with other providers and models. 
    For example, use the `AnthropicChatCompletionBatch` class to run batches with the Anthropic models.
    Check out the [Providers](/concepts/providers/) section to learn more.

```python
from langbatch import AnthropicChatCompletionBatch

batch = AnthropicChatCompletionBatch("path/to/file.jsonl")
batch.start()
```
