# Batch

## Initialize a Batch

You can initialize a batch by passing the path to a JSONL file. File should be in OpenAI format.

```python
from langbatch import OpenAIChatCompletionBatch

batch = OpenAIChatCompletionBatch("data.jsonl")
print(batch.id)
```

You can also pass a list of requests to the `create` method to create a batch.

```python
batch = OpenAIChatCompletionBatch.create([
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of the moon?"}
    ]},
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of the moon?"}
        ]
    }
])
```

!!! note
    When you initialize a batch, it is not started automatically. You need to call the `start` method to start the batch job.

## Start a Batch

You can start a batch by calling the `start` method.

```python
batch.start()
```

## Get Batch Status

You can get the status of a batch by calling the `get_status` method.

```python
batch.get_status()
```

## Retry Batch

Incase of any failure with batch job due to rate limits, exceeded 24h wait time or other issues, you can retry the same batch by calling the `retry` method.

```python
batch.retry()
```

!!! note
    A batch instance in LangBatch can be retried multiple times. It will internally create multiple batch jobs until the batch is successful.

## Get Batch Results

You can get the results of a completed batch by calling the `get_results` method.

```python
successful_results, unsuccessful_results = batch.get_results()

for result in successful_results:
    print(result)
```

## Get Batch Results File

You can get the results file of a completed batch by calling the `get_results_file` method. This file will be in OpenAI Batch Result JSONL format.

```python
file_path = batch.get_results_file()

with open(file_path, "r") as file:
    results = file.readlines()

for result in results:
    print(result)
```

## Get Unsuccessful Requests

You can get the unsuccessful requests of a batch by calling the `get_unsuccessful_requests` method. This will be useful to retry failed requests or to debug the issues with the requests.

```python
unsuccessful_requests = batch.get_unsuccessful_requests()

for request in unsuccessful_requests:
    print(request)
```

## Get Requests by Custom IDs

You can get the requests of a batch by passing the custom ids to the `get_requests_by_custom_ids` method. This will be useful to get the requests to debug the issues with the requests.

```python
custom_ids = ["custom_id_1", "custom_id_2"]
requests = batch.get_requests_by_custom_ids(custom_ids)

for request in requests:
    print(request)
```

## Save Batch

You can save a batch by calling the `save` method. This will be useful to keep track of created batches and resume the batch job later.

```python
# save batch
batch.save()

# Save Batch to Custom Storage
storage = CustomStorage()
batch = OpenAIChatCompletionBatch("data.jsonl", storage=storage)
batch.save()
```

!!! note
    By default, File based storage will be used to save the batch. You can also pass a custom storage instance when initializing the batch.

## Load Batch

You can load a batch by calling the `load` method. This will be useful to resume the batch job later.

```python
# load batch
batch = OpenAIChatCompletionBatch.load("batch_id")

# Load Batch from Custom Storage
storage = CustomStorage()
batch = OpenAIChatCompletionBatch.load("batch_id", storage=storage)
```
