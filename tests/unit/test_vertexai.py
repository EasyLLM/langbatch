import os
from pathlib import Path
import json
import time
from datetime import datetime
import pytest
import jsonlines
import vertexai
from google.oauth2 import service_account
from langbatch.vertexai import VertexAIChatCompletionBatch
from langbatch.batch_storages import FileBatchStorage
from tests.unit.fixtures import test_data_file, temp_dir

GCP_PROJECT = os.environ.get('GCP_PROJECT')
GCP_LOCATION = os.environ.get('GCP_LOCATION')
BIGQUERY_INPUT_DATASET = os.environ.get('BIGQUERY_INPUT_DATASET')
BIGQUERY_OUTPUT_DATASET = os.environ.get('BIGQUERY_OUTPUT_DATASET')
VERTEX_AI_COMPLETED_BATCH_ID = os.environ.get('VERTEX_AI_COMPLETED_BATCH_ID')
VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID = os.environ.get('VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID')

def setup_module():
    SERVICE_ACCOUNT_KEY_FILE = os.environ.get('GCP_SERVICE_ACCOUNT_KEY_FILE')
    credentials = service_account.Credentials.from_service_account_file(filename=SERVICE_ACCOUNT_KEY_FILE)
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION, credentials = credentials)

@pytest.fixture
def vertexai_batch(test_data_file: str):
    batch = VertexAIChatCompletionBatch(
        file=test_data_file,
        model_name="gemini-1.5-flash-002",
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET)
    return batch

def test_vertexai_batch_init(vertexai_batch: VertexAIChatCompletionBatch):
    assert vertexai_batch.model_name == "gemini-1.5-flash-002"
    assert vertexai_batch.gcp_project == GCP_PROJECT
    assert vertexai_batch.bigquery_input_dataset == BIGQUERY_INPUT_DATASET
    assert vertexai_batch.bigquery_output_dataset == BIGQUERY_OUTPUT_DATASET

    # check if the platform_batch_id is None
    assert vertexai_batch.platform_batch_id is None

    # check if the file is a file
    assert Path(vertexai_batch._file).is_file()

    # check if the file can be read as a jsonl file
    original_data = vertexai_batch._get_requests()
    for req in original_data:
        assert isinstance(req, dict)

def test_vertexai_batch_create(test_data_file):
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append(req)
    
    batch = VertexAIChatCompletionBatch.create_from_requests(requests, batch_kwargs={
        "model_name": "gemini-1.5-flash-002",
        "gcp_project": GCP_PROJECT,
        "bigquery_input_dataset": BIGQUERY_INPUT_DATASET,
        "bigquery_output_dataset": BIGQUERY_OUTPUT_DATASET
    })
    
    assert isinstance(batch, VertexAIChatCompletionBatch)
    assert batch.model_name == "gemini-1.5-flash-002"
    assert batch.gcp_project == GCP_PROJECT
    assert batch.bigquery_input_dataset == BIGQUERY_INPUT_DATASET
    assert batch.bigquery_output_dataset == BIGQUERY_OUTPUT_DATASET
    
    # Check if the file is created and contains the correct number of requests
    batch_file_requests = []
    with jsonlines.open(batch._file) as reader:
        for req in reader:
            batch_file_requests.append(req)
    assert len(batch_file_requests) == len(requests)
    for req, batch_req in zip(requests, batch_file_requests):
        assert req == batch_req

def test_vertexai_batch_save_and_load(vertexai_batch: VertexAIChatCompletionBatch, temp_dir):
    storage = FileBatchStorage(temp_dir)
    
    # Save the batch
    vertexai_batch.save(storage=storage)
    
    # Load the batch
    loaded_batch = VertexAIChatCompletionBatch.load(vertexai_batch.id, storage=storage)
    
    assert loaded_batch.id == vertexai_batch.id
    assert loaded_batch.model_name == vertexai_batch.model_name
    assert loaded_batch.gcp_project == vertexai_batch.gcp_project
    assert loaded_batch.bigquery_input_dataset == vertexai_batch.bigquery_input_dataset
    assert loaded_batch.bigquery_output_dataset == vertexai_batch.bigquery_output_dataset
    assert loaded_batch.platform_batch_id == vertexai_batch.platform_batch_id

    vertexai_batch.platform_batch_id = "new_platform_batch_id"
    vertexai_batch.save(storage=storage)
    loaded_batch = VertexAIChatCompletionBatch.load(vertexai_batch.id, storage=storage)
    assert loaded_batch.platform_batch_id == vertexai_batch.platform_batch_id

def test_vertexai_batch_get_status(vertexai_batch: VertexAIChatCompletionBatch, monkeypatch):
    with pytest.raises(ValueError, match="Batch not started"):
        vertexai_batch.get_status()

    vertexai_batch.platform_batch_id = VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID
    assert vertexai_batch.get_status() == "completed"
    
    vertexai_batch.platform_batch_id = "platform_batch_id-1"
    monkeypatch.setattr(vertexai_batch, 'get_status', lambda: "in_progress")
    assert vertexai_batch.get_status() == "in_progress"

def test_vertexai_batch_convert_request(test_data_file):
    batch = VertexAIChatCompletionBatch(
        file=test_data_file,
        model_name="gemini-1.5-flash-002",
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET
    )
    
    request = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "model": "gemini-flash-1.5-002",
        "temperature": 0.7,
        "max_tokens": 100,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    }
    
    converted = json.loads(batch._convert_request(request))
    
    assert "contents" in converted
    assert "systemInstruction" in converted
    assert "tools" in converted
    assert "generationConfig" in converted
    
    assert converted["systemInstruction"]["role"] == "system"
    assert len(converted["contents"]) == 1
    assert converted["contents"][0]["role"] == "user"
    assert len(converted["tools"]) == 1
    assert converted["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert converted["generationConfig"]["temperature"] == 0.7
    assert converted["generationConfig"]["maxOutputTokens"] == 100

def test_vertexai_batch_convert_response(test_data_file):
    batch = VertexAIChatCompletionBatch(
        file=test_data_file,
        model_name="gemini-1.5-flash-002",
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET
    )
    
    response = {
        "custom_id": "test_id",
        "status": "",
        "processed_time": datetime.fromisoformat("1997-08-06T00:00:00Z"),
        "response": json.dumps({
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello! How can I assist you today?"}]},
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "totalTokenCount": 18
            }
        })
    }
    
    converted = batch._convert_response(response)
    
    assert converted["id"] == "test_id"
    assert converted["custom_id"] == "test_id"
    assert converted["response"]["body"]["choices"][0]["message"]["role"] == "assistant"
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == "Hello! How can I assist you today?"
    assert converted["response"]["body"]["usage"]["prompt_tokens"] == 10
    assert converted["response"]["body"]["usage"]["completion_tokens"] == 8
    assert converted["response"]["body"]["usage"]["total_tokens"] == 18

@pytest.mark.slow
def test_vertexai_batch_start(vertexai_batch: VertexAIChatCompletionBatch):
    vertexai_batch.start()
    assert vertexai_batch.platform_batch_id is not None

    time.sleep(5)
    assert vertexai_batch.get_status() == "in_progress"

def test_vertexai_batch_get_results(vertexai_batch: VertexAIChatCompletionBatch, monkeypatch):
    monkeypatch.setattr(vertexai_batch, 'id', VERTEX_AI_COMPLETED_BATCH_ID)
    successful_results, unsuccessful_results = vertexai_batch.get_results()
    
    assert len(successful_results) > 0
    assert len(unsuccessful_results) == 0

    for successful_result in successful_results:
        assert successful_result["custom_id"] is not None
        assert successful_result["choices"] is not None
        assert len(successful_result["choices"]) > 0
        assert successful_result["choices"][0]["message"]["content"] is not None
