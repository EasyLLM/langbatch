# Batch Dispatcher

## Why do we need Batch Dispatcher?

Batch Dispatcher is responsible for creating batches from incoming stream of requests and dispatching them to the batch handler in a balanced manner. This is useful in situations where you need to setup a API service and listen to incoming requests.

Also, we need to keep track of all the incoming requests and maintain a queue.

## Initialize a Batch Dispatcher

You can initialize a batch dispatcher by passing the batch handler, request queue, queue threshold, time threshold, time interval, and request kwargs.

```python
from langbatch import BatchHandler
from langbatch.request_queues import InMemoryRequestQueue
from langbatch import BatchDispatcher

# Create a batch handler
batch_handler = BatchHandler(
    batch_process_func=process_batch,
    batch_type=OpenAIChatCompletionBatch
)

# Create a request queue
request_queue = InMemoryRequestQueue()

# Create a batch dispatcher
batch_dispatcher = BatchDispatcher(
    batch_handler=batch_handler,
    queue=request_queue,
    queue_threshold=50000, # 50000 requests
    time_threshold=3600 * 2, # 2 hours
    time_interval=600, # 10 minutes,
    requests_type="partial",
    request_kwargs=request_kwargs
)
```

Here, we are initializing a batch dispatcher with a InMemoryRequestQueue. 

`queue_threshold` is the maximum number of requests that can be added to the queue. If the queue threshold is reached, then the requests will be converted into a batch and sent to the batch handler.

`time_threshold` is the maximum time interval for which a request can be waited in queue. Even if the queue threshold is not reached. Once the time threshold is reached, the requests in the queue will be converted into a batch and sent to the batch handler.

`time_interval` is the time interval for which the queue will be checked for the queue threshold and time threshold.

`requests_type` is the type of requests that will be added to the queue. It can be "partial" or "full". If it is 'partial', [Batch.create](/references/ChatCompletion/#langbatch.ChatCompletionBatch.ChatCompletionBatch.create) method is used to create the batch, and if it is 'full', [Batch.create_from_requests](/references/Batch/#langbatch.Batch.Batch.create_from_requests) method is used.

`request_kwargs` is the kwargs that will be passed to the [Batch.create](/references/ChatCompletion/#langbatch.ChatCompletionBatch.ChatCompletionBatch.create) method in Batch class to create a batch. Ex. temperature, max_tokens, etc. Used when `requests_type` is 'partial'.

## Run the Batch Dispatcher

```python
asyncio.create_task(batch_dispatcher.run())
```

This will start a background task that will run indefinitely until the program is terminated.

## Add Requests to the Queue

```python
# Add multiple requests to the queue
await request_queue.add_requests(requests)
```

## Redis Request Queue

You can also use RedisRequestQueue to add requests to the queue. With RedisRequestQueue, 
* you can add requests to the queue in a persisted, distributed manner
* it can be shared across multiple processes and machines to add requests to the queue.

```python
from langbatch.request_queues import RedisRequestQueue
import redis

REDIS_URL = os.environ.get('REDIS_URL')
redis_client = redis.from_url(REDIS_URL)

request_queue = RedisRequestQueue(
    redis_client=redis_client,
    queue_name='gemini_requests'
)
```

## Custom Request Queue

You can also implement your own request queue by implementing the `RequestQueue`.

```python
from langbatch.request_queues import RequestQueue

class CustomRequestQueue(RequestQueue):
    pass
```