import pytest
import json
import pickle
from pathlib import Path
from pydantic import BaseModel

from langbatch.batch_storages import BatchStorage, FileBatchStorage, _is_json_serializable
from tests.unit.fixtures import test_data_file, temp_dir
from langbatch.errors import BatchStorageError

@pytest.fixture(params=[FileBatchStorage])
def batch_storage(request, temp_dir: str) -> BatchStorage:
    storage_class = request.param
    if storage_class == FileBatchStorage:
        return storage_class(directory=temp_dir)
    else:
        return storage_class()
    
class TestObject(BaseModel):
    integer: int
    string: str

def test_is_json_serializable():
    # Test simple types
    assert _is_json_serializable({"a": 1, "b": "test"}) == True
    assert _is_json_serializable([1, 2, "test"]) == True
    assert _is_json_serializable("test") == True
    assert _is_json_serializable(123) == True

    # Test complex objects
    test_object = TestObject(integer=1, string="test")
    assert _is_json_serializable({"object": test_object}) == False

def test_batch_storage_save_and_load_json(batch_storage: BatchStorage, test_data_file: Path):
    """Test saving and loading with JSON-serializable metadata"""
    # Test saving
    meta_data = {
        "platform_batch_id": "xyz",
        "model": "gemini-2.0-flash-001"
    }
    batch_storage.save('test-json', test_data_file, meta_data)

    # Test loading
    data_file, meta_file = batch_storage.load('test-json')

    # Check if files exist
    assert data_file.exists()
    assert meta_file.exists()
    assert meta_file.suffix == '.json'  # Should use JSON for simple metadata

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

def test_batch_storage_save_and_load_pickle(batch_storage: BatchStorage, test_data_file: Path):
    """Test saving and loading with pickle-requiring metadata"""
    # Create test data with non-JSON-serializable objects
    test_object = TestObject(integer=1, string="test")
    meta_data = {
        "platform_batch_id": "xyz",
        "object": test_object
    }
    
    batch_storage.save('test-pickle', test_data_file, meta_data)

    # Test loading
    data_file, meta_file = batch_storage.load('test-pickle')

    # Check if files exist
    assert data_file.exists()
    assert meta_file.exists()
    assert meta_file.suffix == '.pkl'  # Should use pickle for complex metadata

    # Check metadata
    with open(meta_file, 'rb') as f:
        loaded_meta = pickle.load(f)
    assert isinstance(loaded_meta["object"], TestObject)
    assert loaded_meta["object"].integer == test_object.integer
    assert loaded_meta["object"].string == test_object.string
    assert loaded_meta["platform_batch_id"] == meta_data["platform_batch_id"]

    # Check data
    with open(data_file, 'r') as f:
        loaded_data = [json.loads(line) for line in f]
        
    with open(test_data_file, mode='r') as f:
        test_data = [json.loads(line) for line in f]
    assert loaded_data == test_data

def test_batch_storage_nonexistent_batch(batch_storage: BatchStorage):
    with pytest.raises(BatchStorageError, match="Batch with id nonexistent_batch not found"):
        batch_storage.load("nonexistent_batch")

def test_batch_storage_directory_creation(temp_dir: str):
    FileBatchStorage(directory=str(temp_dir))
    assert (Path(temp_dir) / "saved_batches").exists()

def test_batch_storage_overwrite_existing_batch(batch_storage: BatchStorage, test_data_file: Path):
    """Test that saving a batch with an existing ID overwrites the existing data."""
    batch_id = 'test-1'
    
    # Initial save with JSON
    meta_data = {"platform_batch_id": "xyz"}
    batch_storage.save(batch_id, test_data_file, meta_data)
    _, meta_file = batch_storage.load(batch_id)
    assert meta_file.suffix == '.json'

    # Overwrite with pickle-requiring data
    test_object = TestObject(integer=1, string="test")
    new_meta_data = {
        "platform_batch_id": "xyz",
        "object": test_object
    }
    batch_storage.save(batch_id, test_data_file, new_meta_data)
    
    # Check if new metadata is saved with pickle
    _, meta_file = batch_storage.load(batch_id)
    assert meta_file.suffix == '.pkl'
    with open(meta_file, 'rb') as f:
        loaded_meta = pickle.load(f)
    assert isinstance(loaded_meta["object"], TestObject)
    assert loaded_meta["object"].integer == test_object.integer
    assert loaded_meta["object"].string == test_object.string

def test_file_batch_storage_partial_save(batch_storage: FileBatchStorage):
    """Test the behavior when only metadata is saved without data file."""
    # Manually save metadata without data file
    meta_file_path = batch_storage.saved_batches_directory / f"test-1.json"
    with open(meta_file_path, 'w') as f:
        json.dump({"platform_batch_id": "xyz"}, f)

    # Attempt to load should raise an error due to missing data file
    with pytest.raises(BatchStorageError, match=f"Batch with id test-1 not found"):
        batch_storage.load("test-1")