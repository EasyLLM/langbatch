from pathlib import Path
import json

import pytest
import jsonlines

from langbatch.openai import OpenAIChatCompletionBatch
from langbatch.batch_storages import FileBatchStorage
from tests.unit.fixtures import *
from langbatch.errors import BatchValidationError, BatchStorageError

def test_init(batch: OpenAIChatCompletionBatch):
    # check if the id is not None
    assert batch.id is not None

    # check if the platform_batch_id is None
    assert batch.platform_batch_id is None

    # check if the file is a file
    assert Path(batch._file).is_file()

    # check if the file can be read as a jsonl file
    original_data = batch._get_requests()
    for req in original_data:
        assert isinstance(req, dict)

cases = [
    ('chat_completion_batch_invalid.jsonl', BatchValidationError, r"Invalid requests: \[.+\]"),
    ('chat_completion_batch_empty.jsonl', BatchValidationError, r"No requests found in the batch file"),
]
@pytest.mark.parametrize(
    'test_data_file, expected_error, error_message', 
    cases, 
    indirect=['test_data_file']
)
def test_init_error(test_data_file, expected_error, error_message):
    with pytest.raises(expected_error, match=error_message):
        OpenAIChatCompletionBatch(test_data_file)

@pytest.mark.parametrize('test_data_file', ['chat_completion_partial_requests.jsonl'], indirect=True)
def test_create(test_data_file):
    # load the requests from the file
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append(req["messages"])
    
    # create the batch
    request_kwargs = {"model": "gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": 500
    }
    batch = OpenAIChatCompletionBatch.create(requests, request_kwargs)
    assert batch.id is not None
    assert batch.platform_batch_id is None
    assert batch._file is not None
    assert Path(batch._file).is_file()

    batch_requests = batch._get_requests()
    assert len(batch_requests) > 0
    assert len(batch_requests) == len(requests)

    # check if the batch is created correctly
    for index, req in enumerate(batch_requests):
        assert 'custom_id' in req
        assert 'method' in req
        assert 'url' in req
        assert 'body' in req

        assert "messages" in req['body']
        assert len(req['body']['messages']) > 0
        assert req['body']['messages'] == requests[index]
        assert req['body']['model'] == request_kwargs['model']
        assert req['body']['temperature'] == request_kwargs['temperature']
        assert req['body']['max_tokens'] == request_kwargs['max_tokens']
        
        assert batch._validate_request(req["body"]) is None

@pytest.mark.parametrize('test_data_file', ['chat_completion_batch.jsonl'], indirect=True)
def test_create_from_requests(test_data_file):
    # load the requests from the file
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append(req)
    
    # create the batch
    batch = OpenAIChatCompletionBatch.create_from_requests(requests)
    batch_requests = batch._get_requests()

    # check if the batch is created correctly
    for index, req in enumerate(batch_requests):
        assert req == requests[index]
        assert batch._validate_request(req["body"]) is None
            

    assert len(batch_requests) > 0
    assert len(batch_requests) == len(requests)

def test_save(
    batch: OpenAIChatCompletionBatch,
    temp_dir,
    monkeypatch
):
    # save a batch to the storage
    storage = FileBatchStorage(temp_dir)
    batch.save(storage=storage)

    # check if the batch is saved to the storage
    jsonl_file_path, json_file_path = storage.load(batch.id)
    assert json_file_path.is_file()
    assert jsonl_file_path.is_file()

    # check if the platform_batch_id is correctly saved
    with open(json_file_path, 'r') as f:
        assert json.load(f)["platform_batch_id"] == batch.platform_batch_id

    # check if the data is correctly saved
    with jsonlines.open(jsonl_file_path) as reader:
        stored_data = [req for req in reader]

    original_data = batch._get_requests()

    assert len(stored_data) == len(original_data)
    for stored_item, original_item in zip(stored_data, original_data):
        assert stored_item == original_item

    # change the platform_batch_id and save the batch to the storage
    monkeypatch.setattr(batch, 'platform_batch_id', 'new_platform_batch_id')
    batch.save(storage=storage)

    # check if the new platform_batch_id is correctly saved
    with open(json_file_path, 'r') as f:
        assert json.load(f)["platform_batch_id"] == 'new_platform_batch_id'

def test_load(
    batch: OpenAIChatCompletionBatch,
    temp_dir,
    monkeypatch
):
    # save a batch to the storage
    storage = FileBatchStorage(temp_dir)
    batch.save(storage=storage)

    # load the batch from the storage 
    loaded_batch = OpenAIChatCompletionBatch.load(
        storage=storage,
        id=batch.id
    )
    
    # check if the platform_batch_id is correctly loaded
    assert loaded_batch.platform_batch_id == batch.platform_batch_id

    # check if the data is correctly loaded
    original_data = batch._get_requests()
    
    with jsonlines.open(loaded_batch._file) as reader:
        loaded_data = [req for req in reader]

    assert len(loaded_data) == len(original_data)
    for loaded_item, original_item in zip(loaded_data, original_data):
        assert loaded_item == original_item

    # change the platform_batch_id and save the batch to the storage
    monkeypatch.setattr(
        batch,
        'platform_batch_id',
        'new_platform_batch_id'
    )
    batch.save(storage=storage)

    # load the batch from the storage
    loaded_batch = OpenAIChatCompletionBatch.load(
        storage=storage,
        id=batch.id
    )

    # check if the platform_batch_id is correctly loaded
    assert loaded_batch.platform_batch_id == batch.platform_batch_id

    # Test loading a batch that doesn't exist
    with pytest.raises(BatchStorageError, match="Batch with id non_existent_batch_id not found"):
        OpenAIChatCompletionBatch.load(storage=storage, id='non_existent_batch_id')

def test_create_results_file_path(batch: OpenAIChatCompletionBatch):
    results_file_path = batch._create_results_file_path()
    
    assert results_file_path is not None
    assert isinstance(results_file_path, Path)

@pytest.mark.parametrize('test_data_file', ['chat_completion_batch_results.jsonl'], indirect=True)
def test_get_results_file(batch: OpenAIChatCompletionBatch, test_data_file, monkeypatch):
    # mock the _create_results_file_path method
    monkeypatch.setattr(batch, '_download_results_file', lambda: test_data_file)

    # download the results file
    results_file = batch.get_results_file()

    # check if the results file is downloaded
    assert results_file.is_file()

    # check if the results file is in OpenAI compatible format
    with jsonlines.open(results_file) as reader:
        results = [res for res in reader]
        assert len(results) > 0

    # mock the _download_results_file method to return None
    monkeypatch.setattr(batch, '_download_results_file', lambda: None)

    # download the results file
    results_file = batch.get_results_file()

    # check if the results file is None
    assert results_file is None

@pytest.mark.parametrize('test_data_file', ['chat_completion_batch_results.jsonl'], indirect=True)
def test_prepare_results(batch: OpenAIChatCompletionBatch, test_data_file, monkeypatch):
    # mock the _download_results_file method
    monkeypatch.setattr(batch, '_download_results_file', lambda: test_data_file)

    # call the _prepare_results method
    process_func = lambda result: {"choices": result['response']['body']['choices']}
    successful_results, unsuccessful_results = batch._prepare_results(process_func)

    assert len(successful_results) > 0
    assert isinstance(successful_results, list)
    assert isinstance(successful_results[0], dict)

    assert len(unsuccessful_results) > 0
    assert isinstance(unsuccessful_results, list)
    assert isinstance(unsuccessful_results[0], dict)

    for successful_result in successful_results:
        assert 'choices' in successful_result
        assert 'custom_id' in successful_result

    for unsuccessful_result in unsuccessful_results:
        assert 'error' in unsuccessful_result
        assert 'custom_id' in unsuccessful_result

@pytest.mark.parametrize('test_data_file', ['chat_completion_batch_results.jsonl'], indirect=True)
def test_get_unsuccessful_requests(batch: OpenAIChatCompletionBatch, test_data_file, monkeypatch):
    # mock the _download_results_file method
    monkeypatch.setattr(batch, '_download_results_file', lambda: test_data_file)

    # call the _get_unsuccessful_requests method
    unsuccessful_requests = batch.get_unsuccessful_requests()

    # check if the unsuccessful requests are in the correct format
    assert unsuccessful_requests is not None
    assert len(unsuccessful_requests) > 0
    assert isinstance(unsuccessful_requests, list)
    assert isinstance(unsuccessful_requests[0], dict)

    custom_ids = [req['custom_id'] for req in unsuccessful_requests]
    assert len(custom_ids) == len(set(custom_ids))
    assert custom_ids == ['req-3', 'req-4']

def test_get_requests_by_custom_ids(batch: OpenAIChatCompletionBatch):
    # call the _get_requests_by_custom_ids method
    custom_ids = ['req-1', 'req-2']
    requests = batch.get_requests_by_custom_ids(custom_ids)

    # check if the requests are in the correct format
    assert len(requests) > 0
    assert len(requests) == 2
    assert isinstance(requests, list)
    assert isinstance(requests[0], dict)

    fetched_custom_ids = [req['custom_id'] for req in requests]
    assert len(fetched_custom_ids) == len(set(fetched_custom_ids))
    assert fetched_custom_ids == custom_ids

    # check with non-existing custom ids
    requests = batch.get_requests_by_custom_ids(['non-existing-custom-id'])
    assert len(requests) == 0

    # check with empty custom ids
    requests = batch.get_requests_by_custom_ids([])
    assert len(requests) == 0
