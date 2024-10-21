import pytest
from pathlib import Path
from langbatch.batch_queues import BatchQueue, FileBatchQueue
from tests.unit.fixtures import test_data_file, temp_dir

@pytest.fixture(params=[FileBatchQueue])
def batch_queue(request, temp_dir: str) -> BatchQueue:
    if request.param == FileBatchQueue:
        return request.param(Path(temp_dir) / "batch_queue.json")
    else:
        raise ValueError(f"Unsupported batch queue: {request.param}")

def test_batch_queue(batch_queue: BatchQueue):
    batch_queue.save({
        "pending": ["batch-3", "batch-4"],
        "processing": ["batch-1", "batch-2"]
    })
    data = batch_queue.load()
    assert data == {
        "pending": ["batch-3", "batch-4"],
        "processing": ["batch-1", "batch-2"]
    }

    batch_queue.save({
        "pending": ["batch-5", "batch-6"],
        "processing": ["batch-3", "batch-4"]
    })
    data = batch_queue.load()
    assert data == {
        "pending": ["batch-5", "batch-6"],
        "processing": ["batch-3", "batch-4"]
    }

    batch_queue.save({
        "pending": ["batch-7", "batch-8"],
        "processing": ["batch-5", "batch-6"]
    })
    data = batch_queue.load()
    assert data == {
        "pending": ["batch-7", "batch-8"],
        "processing": ["batch-5", "batch-6"]
    }