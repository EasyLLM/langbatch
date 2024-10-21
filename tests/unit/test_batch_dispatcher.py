import asyncio
import time
from unittest.mock import AsyncMock
from pathlib import Path

import jsonlines
import pytest

from langbatch.BatchDispatcher import BatchDispatcher
from langbatch.BatchHandler import BatchHandler
from langbatch.request_queues import InMemoryRequestQueue
from langbatch.openai import OpenAIChatCompletionBatch
from langbatch.batch_storages import FileBatchStorage
from langbatch.batch_queues import FileBatchQueue
from tests.unit.fixtures import temp_dir, test_data_file

def process_func(batch):
    return None

@pytest.fixture
def batch_handler(temp_dir):
    return BatchHandler(
        batch_process_func=process_func,
        batch_type=OpenAIChatCompletionBatch,
        batch_storage=FileBatchStorage(temp_dir),
        batch_queue=FileBatchQueue(Path(temp_dir) / "batch_queue.json")
    )

@pytest.fixture
def request_queue():
    return InMemoryRequestQueue()

@pytest.fixture
def requests():
    requests = []
    for _ in range(10000):
        requests.append([{"role": "user", "content": "How can I learn Python?"}])
    return requests

@pytest.fixture
def batch_dispatcher(batch_handler, request_queue):
    return BatchDispatcher(
        batch_handler=batch_handler,
        queue=request_queue,
        queue_threshold=10000,
        time_threshold=120,
        time_interval=60,
        requests_type="partial",
        request_kwargs={"model": "gpt-4o-mini", "temperature": 0.7}
    )

@pytest.mark.asyncio
async def test_run(batch_dispatcher: BatchDispatcher, monkeypatch):
    mock_check_batch_conditions = AsyncMock()
    monkeypatch.setattr(batch_dispatcher, "_check_batch_conditions", mock_check_batch_conditions)
    
    task = asyncio.create_task(batch_dispatcher.run())
    await asyncio.sleep(1)
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    assert mock_check_batch_conditions.called

@pytest.mark.asyncio
async def test_check_batch_conditions_threshold_met(
    batch_dispatcher: BatchDispatcher, 
    request_queue: InMemoryRequestQueue,
    requests
):
    request_queue.add_requests(requests)
    
    await batch_dispatcher._check_batch_conditions()
    
    assert len(request_queue) == 0

@pytest.mark.asyncio
async def test_check_batch_conditions_time_threshold_met(
    batch_dispatcher: BatchDispatcher, 
    request_queue: InMemoryRequestQueue,
    requests
):
    request_queue.add_requests(requests[:1000])
    
    batch_dispatcher.last_batch_time = time.time() - 121
    
    await batch_dispatcher._check_batch_conditions()
    
    assert len(request_queue) == 0

@pytest.mark.asyncio
async def test_dispatch_batch(
    batch_dispatcher: BatchDispatcher,
    requests
):
    batch = OpenAIChatCompletionBatch.create(requests, batch_dispatcher.request_kwargs, {})
    await batch_dispatcher._dispatch_batch(batch)
    assert len(batch_dispatcher.batch_handler.batch_queue.load()["pending"]) == 1

@pytest.mark.asyncio
async def test_create_and_dispatch_batch_partial(
    batch_dispatcher: BatchDispatcher, 
    request_queue: InMemoryRequestQueue,
    requests
):
    request_queue.add_requests(requests)
    
    await batch_dispatcher._create_and_dispatch_batch()
    
    assert len(request_queue) == 0
    assert len(batch_dispatcher.batch_handler.batch_queue.load()["pending"]) == 1

@pytest.mark.asyncio
async def test_create_and_dispatch_batch_full(
    batch_dispatcher: BatchDispatcher, 
    request_queue: InMemoryRequestQueue,
    test_data_file
):
    batch_dispatcher.requests_type = "full"
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for item in reader:
            requests.append(item)
    
    request_queue.add_requests(requests)
    await batch_dispatcher._create_and_dispatch_batch()
    assert len(request_queue) == 0
    assert len(batch_dispatcher.batch_handler.batch_queue.load()["pending"]) == 1
