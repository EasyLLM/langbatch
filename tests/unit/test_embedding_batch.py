import jsonlines
import pytest
import jsonlines

from langbatch.openai import OpenAIEmbeddingBatch
from tests.unit.fixtures import *
from langbatch.errors import BatchValidationError

cases = [
    ('embedding_batch_invalid.jsonl', BatchValidationError, r"Invalid requests: \[.+\]"),
    ('embedding_batch_empty.jsonl', BatchValidationError, r"No requests found in the batch file"),
]
@pytest.mark.parametrize(
    'test_data_file, expected_error, error_message', 
    cases, 
    indirect=['test_data_file']
)
def test_init_error(test_data_file, expected_error, error_message):
    with pytest.raises(expected_error, match=error_message):
        OpenAIEmbeddingBatch(test_data_file)

@pytest.mark.parametrize('test_data_file', ['embedding_partial_requests.jsonl'], indirect=True)
def test_create(test_data_file):
    # load the requests from the file
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append(req["input"])
    
    # create the batch
    request_kwargs = {"model": "text-embedding-3-small"}
    batch = OpenAIEmbeddingBatch.create(requests, request_kwargs)
    batch_requests = batch._get_requests()

    assert len(batch_requests) > 0
    assert len(batch_requests) == len(requests)

    # check if the batch is created correctly
    for index, req in enumerate(batch_requests):
        assert 'custom_id' in req
        assert 'method' in req
        assert 'url' in req
        assert 'body' in req

        assert "input" in req['body']
        assert len(req['body']['input']) > 0
        assert req['body']['input'] == requests[index]
        assert req['body']['model'] == request_kwargs['model']
        
        assert batch._validate_request(req["body"]) is None

@pytest.mark.parametrize('test_data_file', ['embedding_batch.jsonl'], indirect=True)
def test_create_from_requests(test_data_file):
    # load the requests from the file
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append(req)
    
    # create the batch
    batch = OpenAIEmbeddingBatch.create_from_requests(requests)
    batch_requests = batch._get_requests()

    # check if the batch is created correctly
    for index, req in enumerate(batch_requests):
        assert req == requests[index]
        assert batch._validate_request(req["body"]) is None
            

    assert len(batch_requests) > 0
    assert len(batch_requests) == len(requests)

@pytest.mark.parametrize('test_data_file', ['embedding_batch_results.jsonl'], indirect=True)
def test_get_results(embedding_batch: OpenAIEmbeddingBatch, test_data_file, monkeypatch):
    # mock the _download_results_file method
    monkeypatch.setattr(embedding_batch, '_download_results_file', lambda: test_data_file)

    # call the _prepare_results method
    successful_results, unsuccessful_results = embedding_batch.get_results()

    assert len(successful_results) > 0
    assert isinstance(successful_results, list)
    assert isinstance(successful_results[0], dict)

    assert len(unsuccessful_results) == 0

    for successful_result in successful_results:
        assert 'embedding' in successful_result
        assert 'custom_id' in successful_result

    for unsuccessful_result in unsuccessful_results:
        assert 'error' in unsuccessful_result
        assert 'custom_id' in unsuccessful_result
    