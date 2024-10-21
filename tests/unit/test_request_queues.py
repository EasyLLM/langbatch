import pytest
import os
import redis
import time

from langbatch.request_queues import InMemoryRequestQueue, RedisRequestQueue, RequestQueue

# Define the test data
TEST_REQUESTS = [
    [
        {"role": "user", "content": "How can I learn Python?"}
    ],
    [
        {"role": "user", "content": "Who is the first president of the United States?"},
        {"role": "assistant", "content": "George Washington"},
        {"role": "user", "content": "Second?"}
    ]
]

@pytest.fixture(params=[InMemoryRequestQueue, RedisRequestQueue])
def request_queue(request) -> RequestQueue:
    queue_class = request.param
    if queue_class == RedisRequestQueue:
        # You might need to add Redis connection parameters here
        redis_client = redis.from_url(os.environ.get('REDIS_URL'))
        return RedisRequestQueue(
            redis_client=redis_client,
            queue_name=str(int(time.time()))
        )
    return queue_class()

def test_request_queue(request_queue: RequestQueue):
    # Test getting requests when queue is empty
    requests = request_queue.get_requests(3)
    assert len(requests) == 0
    # Add requests
    request_queue.add_requests(TEST_REQUESTS)

    # Test queue length
    assert len(request_queue) == 2

    # Test getting requests
    requests = request_queue.get_requests(2)
    assert len(requests) == 2
    assert requests[0] == TEST_REQUESTS[0]
    assert requests[1] == TEST_REQUESTS[1]

    # Test queue is empty after getting all requests
    assert len(request_queue) == 0

    # Test getting more requests than available
    request_queue.add_requests(TEST_REQUESTS)
    requests = request_queue.get_requests(3)
    assert len(requests) == 2

    # Test getting partial requests
    request_queue.add_requests(TEST_REQUESTS)
    requests = request_queue.get_requests(1)
    assert len(requests) == 1
    assert requests[0] == TEST_REQUESTS[0]
    requests = request_queue.get_requests(1)
    assert len(requests) == 1
    assert requests[0] == TEST_REQUESTS[1]

    # Test getting requests when queue is empty after some operations
    requests = request_queue.get_requests(3)
    assert len(requests) == 0