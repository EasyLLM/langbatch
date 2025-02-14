# Batch Handler

## Why do we need Batch Handler?

When we start a batch job in OpenAI or in other providers, it can fail due to various reasons like rate limits, quota limits, etc. We need a mechanism to check the status of the batch periodically and retry the failed batches. And we need to process the completed batches. Also we may need to keep track of the batch jobs and make sure of successful completion of all the batch jobs. 

And due to the Quota and Rate Limits, we need to handle the batches in a queue manner to avoid Rate Limit errors.

BatchHandler is designed to handle all these things in a production environment.

## Initialize a Batch Handler

You can initialize a batch handler by passing the batch process function and the batch type.

```python
from langbatch import BatchHandler

# Create a batch handler process
batch_handler = BatchHandler(
    batch_process_func=process_batch,
    batch_type=OpenAIChatCompletionBatch
)
asyncio.create_task(batch_handler.run())
```

batch_process_func is the function that will be called to process the batch. It should accept batch as the argument. You can put the logic to process the batch in this function.

```python
def process_batch(batch: OpenAIChatCompletionBatch):
    successful_results, _ = batch.get_results()
    for result in successful_results:
        # Process the result
        pass
```

## Add Batches to the Queue

You can add batches to the queue by calling the `add_batch` method.

```python
await batch_handler.add_batch("55d506ef-2a1f-4ca1-9c6c-3fd2415c83f7")
await batch_handler.add_batch("10d71a17-6e29-4b1a-ba3d-245bd7cdf4f0")
```

## Wait Time

Wait time is the time in seconds to wait for processing the next set of batches in time intervals. It is used to avoid Rate Limit errors.

```python
batch_handler = BatchHandler(
    batch_process_func=process_batch,
    batch_type=OpenAIChatCompletionBatch,
    wait_time=1800 # 30 minutes
)
```

## With Custom Storage

By default, BatchHandler uses `FileBatchQueue` to handle the batch queue. And `FileBatchStorage` to store the batches. You can implement and use custom implemetations of `BatchQueue` and `BatchStorage` for the batch handler by passing them to the `BatchHandler` constructor.

```python
custom_batch_queue = MyCustomBatchQueue()
custom_batch_storage = MyCustomBatchStorage()
batch_handler = BatchHandler(
    batch_process_func=process_batch,
    batch_type=OpenAIChatCompletionBatch,
    batch_queue=custom_batch_queue,
    batch_storage=custom_batch_storage
)
```

## Batch Kwargs

You can pass additional kwargs to the batch process function by passing the `batch_kwargs` parameter to the `BatchHandler` constructor. These kwargs are used to initialize the batch object.

```python
batch_handler = BatchHandler(
    batch_process_func=process_batch,
    batch_type=VertexAIChatCompletionBatch,
    batch_kwargs={
        "model": "gemini-2.0-flash-001",
        "project": "my-project",
        "location": "us-central1",
        "bigquery_input_dataset": "input-dataset",
        "bigquery_output_dataset": "output-dataset"
    }
)
```