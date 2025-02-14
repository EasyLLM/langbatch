import pytest
import json
from pathlib import Path
import jsonlines

from langbatch.batch_storages import BatchStorage, FileBatchStorage
from tests.unit.fixtures import test_data_file, temp_dir

@pytest.fixture(params=[FileBatchStorage])
def batch_storage(request, temp_dir: str) -> BatchStorage:
    storage_class = request.param
    if storage_class == FileBatchStorage:
        return storage_class(directory=temp_dir)
    else:
        return storage_class()

def test_batch_storage_save_and_load(batch_storage: BatchStorage, test_data_file: Path):
    # Test saving
    meta_data = { "platform_batch_id": "xyz"}
    batch_storage.save('test-1', test_data_file, meta_data)

    # Test loading
    data_file, meta_file = batch_storage.load('test-1')

    # Check if files exist
    assert data_file.exists()
    assert meta_file.exists()

    # Check metadata
    with open(meta_file, 'r') as f:
        loaded_meta = json.load(f)
    assert loaded_meta == meta_data

    # Check data
    with open(data_file, 'r') as f:
        loaded_data = [json.loads(line) for line in f]
        
    with open(test_data_file, mode='r') as f:
        test_data = [json.loads(line) for line in f]
    assert loaded_data == test_data

def test_batch_storage_nonexistent_batch(batch_storage: BatchStorage):
    with pytest.raises(ValueError, match="Batch with id nonexistent_batch not found"):
        batch_storage.load("nonexistent_batch")

def test_batch_storage_directory_creation(temp_dir: str):
    FileBatchStorage(directory=str(temp_dir))
    assert (Path(temp_dir) / "saved_batches").exists()

def test_batch_storage_overwrite_existing_batch(batch_storage: BatchStorage, test_data_file: Path):
    """
    Test that saving a batch with an existing ID overwrites the existing data.
    """
    batch_id = 'test-1'
    meta_data = { "platform_batch_id": "xyz"}
    # Initial save
    batch_storage.save(batch_id, test_data_file, meta_data)

    # Modify metadata and data
    new_meta_data = meta_data.copy()
    new_meta_data["model"] = "gemini-2.0-flash-001"

    # read the data and append again to the file
    with open(test_data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    with jsonlines.open(test_data_file, mode='a') as writer:
        writer.write_all(data)

    # Overwrite save
    batch_storage.save(batch_id, test_data_file, new_meta_data)

    # Load and verify
    data_file, meta_file = batch_storage.load(batch_id)

    # Check updated metadata
    with open(meta_file, 'r') as f:
        loaded_meta = json.load(f)
    assert loaded_meta == new_meta_data

    # Check updated data
    with open(data_file, 'r') as f:
        loaded_data = [json.loads(line) for line in f]
    assert loaded_data == data

def test_file_batch_storage_partial_save(batch_storage: FileBatchStorage):
    """
    Test the behavior when only metadata is saved without data file.
    """
    # Manually save metadata without data file
    meta_file_path = batch_storage.saved_batches_directory / f"test-1.json"
    with open(meta_file_path, 'w') as f:
        json.dump({"platform_batch_id": "xyz"}, f)

    # Attempt to load should raise an error due to missing data file
    with pytest.raises(ValueError, match=f"Batch with id test-1 not found"):
        batch_storage.load("test-1")