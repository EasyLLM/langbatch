import asyncio
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import pytest

from langbatch.BatchHandler import BatchHandler, BatchStatus
from langbatch.Batch import Batch
from langbatch.openai import OpenAIChatCompletionBatch
from langbatch.batch_storages import FileBatchStorage
from langbatch.batch_queues import FileBatchQueue
from tests.unit.fixtures import temp_dir, batch

@pytest.fixture
def mock_process_func():
    return AsyncMock()

@pytest.fixture
def batch_handler(temp_dir, mock_process_func):
    return BatchHandler(
        batch_process_func=mock_process_func,
        batch_type=OpenAIChatCompletionBatch,
        batch_storage=FileBatchStorage(temp_dir),
        batch_queue=FileBatchQueue(Path(temp_dir) / "batch_queue.json")
    )

@pytest.mark.asyncio
async def test_run(batch_handler):
    # Mock methods
    batch_handler.process_completed_batch = AsyncMock()
    batch_handler._handle_failed_or_expired_batch = AsyncMock(return_value=True)
    batch_handler.cancel_batch = AsyncMock()
    batch_handler.start_batch = AsyncMock()

    # Mock batch loading and status
    mock_batch = MagicMock()
    mock_batch.get_status = MagicMock(side_effect=[
        BatchStatus.COMPLETED.value,
        BatchStatus.FAILED.value,
        BatchStatus.CANCELLED.value
    ])
    batch_handler.batch_type.load = MagicMock(return_value=mock_batch)

    # Set up queues
    batch_handler.queues = {
        "processing": ["batch1", "batch2", "batch3"],
        "pending": ["batch4", "batch5"]
    }

    # Reduce wait time for testing
    batch_handler.wait_time = 0.1

    # Run the method for a short time
    task = asyncio.create_task(batch_handler.run())
    await asyncio.sleep(0.3)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Assertions
    assert batch_handler.process_completed_batch.called
    assert batch_handler._handle_failed_or_expired_batch.called
    assert batch_handler.cancel_batch.called
    assert batch_handler.start_batch.call_count == 2  # Should start 2 pending batches

    # Check if the method respects the limit of 4 batches processed per cycle
    total_processed = (batch_handler.process_completed_batch.call_count +
                       batch_handler._handle_failed_or_expired_batch.call_count +
                       batch_handler.start_batch.call_count)
    assert total_processed <= 4

@pytest.mark.asyncio
async def test_add_batch(batch_handler: BatchHandler, batch: Batch):
    batch.save(batch_handler.batch_storage)
    await batch_handler.add_batch(batch.id)
    assert len(batch_handler.batch_queue.load()["pending"]) == 1
    assert batch.id in batch_handler.queues["pending"]

    batch.id = "new_id"
    batch.save(batch_handler.batch_storage)
    await batch_handler.add_batch(batch.id)
    assert len(batch_handler.batch_queue.load()["pending"]) == 2
    assert batch.id in batch_handler.queues["pending"]

@pytest.mark.asyncio
async def test_process_completed_batch(batch_handler: BatchHandler, batch: Batch, caplog):
    batch_handler.queues = {
        "pending": [],
        "processing": [batch.id],
    }

    await batch_handler.process_completed_batch(batch)
    batch_handler.batch_process_func.assert_called_once_with(batch)
    assert len(batch_handler.queues["processing"]) == 0

    batch_handler.queues = {
        "pending": [],
        "processing": [],
    }

    await batch_handler.process_completed_batch(batch)
    assert batch_handler.batch_process_func.call_count == 1
    assert f"Completed batch {batch.id} not found in processing queue" in caplog.text

    batch_handler.queues = {
        "pending": [],
        "processing": [batch.id],
    }
    batch_handler._save_queues = MagicMock(side_effect=Exception("Test error"))
    await batch_handler.process_completed_batch(batch)
    assert batch_handler.batch_process_func.call_count == 2
    assert f"Error processing completed batch {batch.id}" in caplog.text

@pytest.mark.asyncio
async def test_start_batch(batch_handler: BatchHandler, batch: Batch):
    batch_handler.queues = {
        "pending": [batch.id],
        "processing": []
    }
    
    batch.start = AsyncMock()
    await batch_handler.start_batch(batch)
    
    assert batch.id not in batch_handler.queues["pending"]
    assert batch.id in batch_handler.queues["processing"]
    batch.start.assert_called_once()

@pytest.mark.asyncio
async def test_start_batch_not_in_pending(batch_handler: BatchHandler, batch: Batch, caplog):
    batch_handler.queues = {
        "pending": [],
        "processing": []
    }
    
    batch.start = AsyncMock()
    await batch_handler.start_batch(batch)
    
    assert f"Batch {batch.id} not found in pending queue" in caplog.text
    assert not batch.start.called

@pytest.mark.asyncio
async def test_retry_batch(batch_handler: BatchHandler, batch: Batch):
    batch_handler.queues = {
        "processing": [batch.id]
    }
    
    batch.retry = AsyncMock()
    await batch_handler.retry_batch(batch)
    
    batch.retry.assert_called_once()

@pytest.mark.asyncio
async def test_retry_batch_not_in_processing(batch_handler: BatchHandler, batch: Batch, caplog):
    batch_handler.queues = {
        "processing": []
    }
    batch.retry = AsyncMock()
    await batch_handler.retry_batch(batch)
    
    assert f"Batch {batch.id} not found in processing queue for retry" in caplog.text
    assert not batch.retry.called

@pytest.mark.asyncio
async def test_retry_batch_error(batch_handler: BatchHandler, batch: Batch, caplog):
    batch_handler.queues = {
        "processing": [batch.id]
    }
    batch.retry = MagicMock(side_effect=Exception("Test error"))
    await batch_handler.retry_batch(batch)
    
    assert f"Error retrying batch {batch.id}" in caplog.text
    assert batch.id not in batch_handler.queues["processing"]

@pytest.mark.asyncio
async def test_cancel_batch(batch_handler: BatchHandler):
    batch_id = "test_batch"
    batch_handler.queues = {
        "pending": [batch_id],
        "processing": []
    }
    
    await batch_handler.cancel_batch(batch_id)
    assert batch_id not in batch_handler.queues["pending"]
    assert batch_id not in batch_handler.queues["processing"]

    batch_handler.queues = {
        "pending": [],
        "processing": [batch_id]
    }
    await batch_handler.cancel_batch(batch_id)
    assert batch_id not in batch_handler.queues["pending"]
    assert batch_id not in batch_handler.queues["processing"]

@pytest.mark.asyncio
async def test_cancel_batch_not_found(batch_handler: BatchHandler, caplog):
    batch_id = "non_existent_batch"
    batch_handler.queues = {
        "pending": [],
        "processing": []
    }
    
    await batch_handler.cancel_batch(batch_id)
    assert f"Batch {batch_id} not found in any queue for cancellation" in caplog.text

@pytest.mark.asyncio
async def test_handle_failed_or_expired_batch(batch_handler: BatchHandler, batch: Batch):
    batch_handler.retry_batch = AsyncMock()
    batch_handler.cancel_batch = AsyncMock()
    batch.is_retryable_failure = AsyncMock(return_value=True)
    
    # Test retryable failure
    batch_handler.queues = {"processing": [batch.id]}
    result = await batch_handler._handle_failed_or_expired_batch(batch, BatchStatus.FAILED)
    assert result is True
    batch_handler.retry_batch.assert_called_once_with(batch)
    
    # Test non-retryable failure
    batch_handler.queues = {"processing": [batch.id]}
    batch_handler.retry_batch.reset_mock()
    batch.is_retryable_failure.reset_mock()
    batch.is_retryable_failure.return_value = False
    result = await batch_handler._handle_failed_or_expired_batch(batch, BatchStatus.FAILED)
    assert result is False
    batch_handler.cancel_batch.assert_called_once_with(batch.id)
    
    # Test expired batch
    batch_handler.queues = {"processing": [batch.id]}
    batch_handler.retry_batch.reset_mock()
    result = await batch_handler._handle_failed_or_expired_batch(batch, BatchStatus.EXPIRED)
    assert result is True
    batch_handler.retry_batch.assert_called_once_with(batch)
