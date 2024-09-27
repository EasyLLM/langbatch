# Stream to Batch pipeline with RedisRequestQueue

The previous [guide](./batch_dispatcher_service.md) uses [InMemoryRequestQueue](../references/utils/RequestQueue.md/#langbatch.request_queues.InMemoryRequestQueue) to store the incoming requests. In production, you may want to use a persistent queue to store the incoming requests.

Redis is a popular in-memory data store with persistent storage, and it's easy to set up and use with LangBatch. This guide will show you how to set up a Redis-based [RedisRequestQueue](../references/utils/RequestQueue.md/#langbatch.request_queues.RedisRequestQueue).

Replace
```python
request_queue = InMemoryRequestQueue()
```

with
```python
import os
import redis

REDIS_URL = os.environ.get('REDIS_URL')
redis_client = redis.from_url(REDIS_URL)

request_queue = RedisRequestQueue(redis_client, queue_name='stream_requests')
```