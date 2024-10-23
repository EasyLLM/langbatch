import logging
from typing import List, Any
from collections import deque
from abc import ABC, abstractmethod
import json

class RequestQueue(ABC):
    """
    RequestQueue is an abstract class for request queues.
    Implementations should provide a way to add and retrieve requests.

    Used in `BatchDispatcher` to get requests.

    Usage:
    ```python
    request_queue = InMemoryRequestQueue()
    request_queue.add_requests([
        [
            {"role": "user", "content": "How can I learn Python?"}
        ],
        [
            {"role": "user", "content": "Who is the first president of the United States?"},
            {"role": "assistant", "content": "George Washington"},
            {"role": "user", "content": "Second?"}
        ]
    ])
    
    batch_dispatcher = BatchDispatcher(
        batch_handler=batch_handler,
        queue=request_queue
    )

    asyncio.create_task(batch_dispatcher.run())
    ```
    """
    @abstractmethod
    def add_requests(self, requests: List[Any]):
        """
        Add requests to the queue
        """
        pass

    @abstractmethod
    def get_requests(self, count: int) -> List[Any]:
        """
        Get requests from the queue
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

class InMemoryRequestQueue(RequestQueue):
    def __init__(self):
        self.queue = deque()

    def add_requests(self, requests: List[Any]):
        self.queue.extend(requests)
        logging.info(f"Added {len(requests)} requests to queue")

    def get_requests(self, count: int) -> List[Any]:
        if count > len(self.queue):
            count = len(self.queue)
        return [self.queue.popleft() for _ in range(count)]

    def __len__(self):
        return len(self.queue)
    
class RedisRequestQueue(RequestQueue):
    import redis
    """
    RedisRequestQueue is a request queue that uses a Redis list to store requests.

    Usage:
    ```python
    import os
    import redis

    REDIS_URL = os.environ.get('REDIS_URL')
    redis_client = redis.from_url(REDIS_URL)

    request_queue = RedisRequestQueue(redis_client, queue_name='turbo_requests')
    request_queue.add_requests([
        [
            {"role": "user", "content": "How can I learn Python?"}
        ],
        [
            {"role": "user", "content": "Who is the first president of the United States?"},
            {"role": "assistant", "content": "George Washington"},
            {"role": "user", "content": "Second?"}
        ]
    ])
    ```
    """
    def __init__(self, redis_client: redis.Redis, queue_name: str = 'request_queue'):
        self.redis_client = redis_client
        self.queue_name = queue_name

    def add_requests(self, requests: List[Any]):
        count = len(requests)
        for request in requests:
            self.redis_client.rpush(self.queue_name, str(json.dumps(request)))
        logging.debug(f"Added {count} requests to queue.")

    def get_requests(self, count: int) -> List[Any]:
        size = len(self)
        if count > size:
            count = size
        
        if count == 0:
            return []
    
        items = self.redis_client.lpop(self.queue_name, count=count)
        if items is None:
            return []
        
        results = [json.loads(item.decode('utf-8')) for item in items]

        logging.debug(f"Retrieved {len(results)} requests from queue.")
        return results

    def __len__(self):
        length = self.redis_client.llen(self.queue_name)
        return length