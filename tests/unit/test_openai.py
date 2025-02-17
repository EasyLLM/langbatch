import time
import pytest
import os
import jsonlines

from openai import AzureOpenAI
from langbatch.openai import OpenAIChatCompletionBatch, OpenAIEmbeddingBatch
from langbatch.batch_storages import FileBatchStorage
from tests.unit.fixtures import test_data_file, temp_dir, batch, embedding_batch
from tests.unit.test_config import config

OPENAI_COMPLETED_PLATFORM_BATCH_ID = config["openai"]["OPENAI_COMPLETED_PLATFORM_BATCH_ID"]
OPENAI_EMBEDDING_COMPLETED_BATCH_ID = config["openai"]["OPENAI_EMBEDDING_COMPLETED_BATCH_ID"]
AZURE_COMPLETED_PLATFORM_BATCH_ID = config["azure"]["AZURE_COMPLETED_PLATFORM_BATCH_ID"]

@pytest.fixture
def azure_openai_batch(test_data_file: str):
    azure_endpoint = os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION")
    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)
    batch = OpenAIChatCompletionBatch(test_data_file, client)
    return batch

def test_openai_batch_init(batch: OpenAIChatCompletionBatch):
    assert batch.id is not None
    assert batch.platform_batch_id is None
    assert batch._file is not None

    # check if the file can be read as a jsonl file
    original_data = batch._get_requests()
    for req in original_data:
        assert isinstance(req, dict)

def test_openai_batch_create(test_data_file):
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append(req)
    
    batch = OpenAIChatCompletionBatch.create_from_requests(requests)

    assert isinstance(batch, OpenAIChatCompletionBatch)
    
    # Check if the file is created and contains the correct number of requests
    batch_file_requests = []
    with jsonlines.open(batch._file) as reader:
        for req in reader:
            batch_file_requests.append(req)
    assert len(batch_file_requests) == len(requests)
    for req, batch_req in zip(requests, batch_file_requests):
        assert req == batch_req

def test_openai_batch_save_and_load(batch: OpenAIChatCompletionBatch, temp_dir):
    storage = FileBatchStorage(temp_dir)
    
    # Save the batch
    batch.save(storage=storage)
    
    # Load the batch
    loaded_batch = OpenAIChatCompletionBatch.load(batch.id, storage=storage)
    
    assert loaded_batch.id == batch.id
    assert loaded_batch.platform_batch_id == batch.platform_batch_id

    batch.platform_batch_id = "new_platform_batch_id"
    batch.save(storage=storage)
    loaded_batch = OpenAIChatCompletionBatch.load(batch.id, storage=storage)
    assert loaded_batch.platform_batch_id == batch.platform_batch_id

def test_openai_batch_get_status(batch: OpenAIChatCompletionBatch, monkeypatch):
    with pytest.raises(ValueError, match="Batch not started"):
        batch.get_status()

    batch.platform_batch_id = OPENAI_COMPLETED_PLATFORM_BATCH_ID
    assert batch.get_status() == "completed"
    
    batch.platform_batch_id = "platform_batch_id-1"
    monkeypatch.setattr(batch, 'get_status', lambda: "in_progress")
    assert batch.get_status() == "in_progress"

@pytest.mark.slow
def test_openai_batch_start(batch: OpenAIChatCompletionBatch):
    batch.start()
    assert batch.platform_batch_id is not None

    time.sleep(5)
    assert batch.get_status() == "in_progress"

@pytest.mark.slow
def test_azure_openai_batch_start(azure_openai_batch: OpenAIChatCompletionBatch):
    azure_openai_batch.start()
    assert azure_openai_batch.platform_batch_id is not None

    time.sleep(5)
    status = azure_openai_batch.get_status()
    assert status == "in_progress" or status == "validating"

@pytest.mark.slow
def test_openai_embedding_batch_start(embedding_batch: OpenAIEmbeddingBatch):
    embedding_batch.start()
    assert embedding_batch.platform_batch_id is not None

    time.sleep(5)
    assert embedding_batch.get_status() == "in_progress"

def test_openai_batch_get_results(batch: OpenAIChatCompletionBatch, monkeypatch):
    monkeypatch.setattr(batch, 'platform_batch_id', OPENAI_COMPLETED_PLATFORM_BATCH_ID)
    successful_results, unsuccessful_results = batch.get_results()
    
    assert len(successful_results) > 0
    assert len(unsuccessful_results) == 0

    for successful_result in successful_results:
        assert successful_result["custom_id"] is not None
        assert successful_result["choices"] is not None
        assert len(successful_result["choices"]) > 0
        valid_content = successful_result["choices"][0]["message"]["content"] is not None
        valid_tool_calls = successful_result["choices"][0]["message"].get("tool_calls") is not None
        assert valid_content or valid_tool_calls

def test_azure_openai_batch_get_results(azure_openai_batch: OpenAIChatCompletionBatch, monkeypatch):
    monkeypatch.setattr(azure_openai_batch, 'platform_batch_id', AZURE_COMPLETED_PLATFORM_BATCH_ID)
    successful_results, unsuccessful_results = azure_openai_batch.get_results()
    
    assert len(successful_results) > 0
    assert len(unsuccessful_results) == 0
    
    for successful_result in successful_results:
        assert successful_result["custom_id"] is not None
        assert successful_result["choices"] is not None
        assert len(successful_result["choices"]) > 0
        valid_content = successful_result["choices"][0]["message"]["content"] is not None
        valid_tool_calls = successful_result["choices"][0]["message"].get("tool_calls") is not None
        assert valid_content or valid_tool_calls

def test_openai_embedding_batch_get_results(embedding_batch: OpenAIEmbeddingBatch, monkeypatch):
    monkeypatch.setattr(embedding_batch, 'platform_batch_id', OPENAI_EMBEDDING_COMPLETED_BATCH_ID)
    successful_results, unsuccessful_results = embedding_batch.get_results()
    
    assert len(successful_results) > 0
    assert len(unsuccessful_results) == 0

    for successful_result in successful_results:
        assert successful_result["custom_id"] is not None
        assert successful_result["embedding"] is not None