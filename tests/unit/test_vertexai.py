from pathlib import Path
import json
import time
import requests
from subprocess import check_output
from datetime import datetime
import pytest
import jsonlines
import vertexai
from langbatch.vertexai import VertexAIChatCompletionBatch, VertexAILlamaChatCompletionBatch, VertexAIClaudeChatCompletionBatch
from langbatch.batch_storages import FileBatchStorage
from tests.unit.fixtures import test_data_file, temp_dir
from tests.unit.test_config import config
from langbatch.errors import BatchStateError

GCP_PROJECT = config["vertexai"]["GCP_PROJECT"]
GCP_LOCATION = config["vertexai"]["GCP_LOCATION"]
GCP_LOCATION_CLAUDE = config["vertexai"]["GCP_LOCATION_CLAUDE"]
GCP_PROJECT_ID = config["vertexai"]["GCP_PROJECT_ID"]
BIGQUERY_INPUT_DATASET = config["vertexai"]["BIGQUERY_INPUT_DATASET"]
BIGQUERY_OUTPUT_DATASET = config["vertexai"]["BIGQUERY_OUTPUT_DATASET"]
BIGQUERY_INPUT_DATASET_CLAUDE = config["vertexai"]["BIGQUERY_INPUT_DATASET_CLAUDE"]
BIGQUERY_OUTPUT_DATASET_CLAUDE = config["vertexai"]["BIGQUERY_OUTPUT_DATASET_CLAUDE"]
VERTEX_AI_COMPLETED_BATCH_ID = config["vertexai"]["VERTEX_AI_COMPLETED_BATCH_ID"]
VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID = config["vertexai"]["VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID"]
VERTEX_AI_COMPLETED_BATCH_ID_CLAUDE = config["vertexai"]["VERTEX_AI_COMPLETED_BATCH_ID_CLAUDE"]
VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID_CLAUDE = config["vertexai"]["VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID_CLAUDE"]
VERTEX_AI_COMPLETED_BATCH_ID_LLAMA = config["vertexai"]["VERTEX_AI_COMPLETED_BATCH_ID_LLAMA"]
VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID_LLAMA = config["vertexai"]["VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID_LLAMA"]

model = config["vertexai"]["model"]
llama_model = config["vertexai"]["llama_model"]
claude_model = config["vertexai"]["claude_model"]

# run gcloud auth print-access-token to get the token
token = check_output("gcloud auth print-access-token", shell=True).decode().strip()
def request_vertexai(data, model= model, provider="google"):
    location = GCP_LOCATION
    method = "generateContent"
    if provider == "anthropic":
        location = GCP_LOCATION_CLAUDE
        method = "rawPredict"

    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{location}/publishers/{provider}/models/{model}:{method}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

def request_llama(data):
    url = f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/endpoints/openapi/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

def setup_module():
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

@pytest.fixture
def vertexai_batch(test_data_file: str):
    batch = VertexAIChatCompletionBatch(
        file=test_data_file,
        model=model,
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET)
    return batch

@pytest.fixture
def vertexai_llama_batch(test_data_file: str):
    batch = VertexAILlamaChatCompletionBatch(
        file=test_data_file,
        model=llama_model,
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET)
    return batch

@pytest.fixture
def vertexai_claude_batch(test_data_file: str):
    batch = VertexAIClaudeChatCompletionBatch(
        file=test_data_file,
        model=claude_model,
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET_CLAUDE,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET_CLAUDE)
    return batch

def test_vertexai_batch_init(vertexai_batch: VertexAIChatCompletionBatch):
    assert vertexai_batch.model == model
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
        "model": model,
        "gcp_project": GCP_PROJECT,
        "bigquery_input_dataset": BIGQUERY_INPUT_DATASET,
        "bigquery_output_dataset": BIGQUERY_OUTPUT_DATASET
    })
    
    assert isinstance(batch, VertexAIChatCompletionBatch)
    assert batch.model == model
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
    assert loaded_batch.model == vertexai_batch.model
    assert loaded_batch.gcp_project == vertexai_batch.gcp_project
    assert loaded_batch.bigquery_input_dataset == vertexai_batch.bigquery_input_dataset
    assert loaded_batch.bigquery_output_dataset == vertexai_batch.bigquery_output_dataset
    assert loaded_batch.platform_batch_id == vertexai_batch.platform_batch_id

    vertexai_batch.platform_batch_id = "new_platform_batch_id"
    vertexai_batch.save(storage=storage)
    loaded_batch = VertexAIChatCompletionBatch.load(vertexai_batch.id, storage=storage)
    assert loaded_batch.platform_batch_id == vertexai_batch.platform_batch_id

def test_vertexai_batch_get_status(vertexai_batch: VertexAIChatCompletionBatch, monkeypatch):
    with pytest.raises(BatchStateError, match="Batch not started"):
        vertexai_batch.get_status()

    vertexai_batch.platform_batch_id = VERTEX_AI_COMPLETED_PLATFORM_BATCH_ID
    assert vertexai_batch.get_status() == "completed"
    
    vertexai_batch.platform_batch_id = "platform_batch_id-1"
    monkeypatch.setattr(vertexai_batch, 'get_status', lambda: "in_progress")
    assert vertexai_batch.get_status() == "in_progress"

def test_vertexai_batch_convert_request(test_data_file):
    batch = VertexAIChatCompletionBatch(
        file=test_data_file,
        model=model,
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET
    )
    
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "model": model,
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
    
    request = {
        "custom_id": "test_id",
        "body": body
    }

    converted = json.loads(batch._convert_request(request)["request"])
    
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

    for request in batch._get_requests():
        response = request_vertexai(json.loads(batch._convert_request(request)["request"]), provider="google")
        assert len(response["candidates"][0]["content"]["parts"]) > 0

def test_vertexai_batch_convert_response(test_data_file):
    batch = VertexAIChatCompletionBatch(
        file=test_data_file,
        model=model,
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
        valid_content = successful_result["choices"][0]["message"]["content"] is not None
        tool_calls_in_response = "tool_calls" in successful_result["choices"][0]["message"]
        if tool_calls_in_response:
            valid_tool_calls = len(successful_result["choices"][0]["message"]["tool_calls"]) > 0
            assert valid_tool_calls
        else:
            assert valid_content

def test_vertexai_llama_batch_get_results(vertexai_llama_batch: VertexAILlamaChatCompletionBatch, monkeypatch):
    monkeypatch.setattr(vertexai_llama_batch, 'id', VERTEX_AI_COMPLETED_BATCH_ID_LLAMA)
    successful_results, unsuccessful_results = vertexai_llama_batch.get_results()
    
    assert len(successful_results) > 0
    assert len(unsuccessful_results) == 0

    for successful_result in successful_results:
        assert successful_result["custom_id"] is not None
        assert successful_result["choices"] is not None
        assert len(successful_result["choices"]) > 0
        valid_content = successful_result["choices"][0]["message"].get("content") is not None
        tool_calls_in_response = "tool_calls" in successful_result["choices"][0]["message"]
        if tool_calls_in_response:
            valid_tool_calls = len(successful_result["choices"][0]["message"]["tool_calls"]) > 0
            assert valid_tool_calls
        else:
            assert valid_content

def test_vertexai_claude_batch_get_results(vertexai_claude_batch: VertexAIClaudeChatCompletionBatch, monkeypatch):
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION_CLAUDE)

    monkeypatch.setattr(vertexai_claude_batch, 'id', VERTEX_AI_COMPLETED_BATCH_ID_CLAUDE)
    successful_results, unsuccessful_results = vertexai_claude_batch.get_results()
    
    assert len(successful_results) > 0
    assert len(unsuccessful_results) == 0

    for successful_result in successful_results:
        assert successful_result["custom_id"] is not None
        assert successful_result["choices"] is not None
        assert len(successful_result["choices"]) > 0
        valid_content = successful_result["choices"][0]["message"]["content"] is not None
        tool_calls_in_response = "tool_calls" in successful_result["choices"][0]["message"]
        if tool_calls_in_response:
            valid_tool_calls = len(successful_result["choices"][0]["message"]["tool_calls"]) > 0
            assert valid_tool_calls
        else:
            assert valid_content

@pytest.mark.slow
@pytest.mark.parametrize("test_data_file", ["chat_completion_batch_llama.jsonl"], indirect=True)
def test_vertexai_llama_batch_start(test_data_file: str):
    vertexai_llama_batch = VertexAILlamaChatCompletionBatch(
        file=test_data_file,
        model=llama_model,
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET
    )
    vertexai_llama_batch.start()
    assert vertexai_llama_batch.platform_batch_id is not None

    time.sleep(5)
    assert vertexai_llama_batch.get_status() == "in_progress"

@pytest.mark.slow
def test_vertexai_claude_batch_start(test_data_file):
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION_CLAUDE)
    vertexai_claude_batch = VertexAIClaudeChatCompletionBatch(
        file=test_data_file,
        model=claude_model,
        gcp_project=GCP_PROJECT,
        bigquery_input_dataset=BIGQUERY_INPUT_DATASET_CLAUDE,
        bigquery_output_dataset=BIGQUERY_OUTPUT_DATASET_CLAUDE
    )
    vertexai_claude_batch.start()
    assert vertexai_claude_batch.platform_batch_id is not None

    time.sleep(5)
    assert vertexai_claude_batch.get_status() == "in_progress"

def test_vertexai_llama_batch_convert_request(vertexai_llama_batch: VertexAILlamaChatCompletionBatch):
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "model": llama_model,
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    request = {
        "custom_id": "test_id",
        "body": body
    }

    converted = json.loads(vertexai_llama_batch._convert_request(request)["body"])
    
    assert converted["model"] == f"meta/{llama_model}"
    assert converted["messages"] == body["messages"]
    assert converted["temperature"] == 0.7
    assert converted["max_tokens"] == 100

    for request in vertexai_llama_batch._get_requests():
        if request["body"].get("response_format"):
            continue
        request = json.loads(vertexai_llama_batch._convert_request(request)["body"])
        request["model"] = f"meta/{llama_model}"
        response = request_llama(request)

        success_response = "choices" in response
        assert success_response
        if success_response:
            valid_content = response["choices"][0]["message"].get("content") is not None
            tool_calls_in_response = "tool_calls" in response["choices"][0]["message"]
            if tool_calls_in_response:
                valid_tool_calls = len(response["choices"][0]["message"]["tool_calls"]) > 0
                assert valid_tool_calls
            else:
                assert valid_content

def test_vertexai_llama_batch_convert_response(vertexai_llama_batch: VertexAILlamaChatCompletionBatch):
    response = {
        "id": "test_id",
        "custom_id": "test_id",
        "response": json.dumps({
            "id": "test_id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": llama_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How can I help?"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }),
        "error": None
    }
    
    converted = vertexai_llama_batch._convert_response(response)
    
    assert converted["id"] == "test_id"
    assert converted["custom_id"] == "test_id"
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == "Hello! How can I help?"
    assert converted["error"] is None

def test_vertexai_claude_batch_convert_request(vertexai_claude_batch: VertexAIClaudeChatCompletionBatch):
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "model": claude_model,
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
    
    request = {
        "custom_id": "test_id",
        "body": body
    }

    converted = json.loads(vertexai_claude_batch._convert_request(request)["request"])
    
    assert converted["system"] == "You are a helpful assistant."
    assert len(converted["messages"]) == 1
    assert converted["messages"][0]["role"] == "user"
    assert converted["messages"][0]["content"][0]["type"] == "text"
    assert converted["messages"][0]["content"][0]["text"] == "Hello, how are you?"
    assert converted["temperature"] == 0.7
    assert converted["max_tokens"] == 100
    assert len(converted["tools"]) == 1
    assert converted["tools"][0]["name"] == "get_weather"
    assert converted["anthropic_version"] == "vertex-2023-10-16"

    for request in vertexai_claude_batch._get_requests():
        request = json.loads(vertexai_claude_batch._convert_request(request)["request"])
        response = request_vertexai(request, provider="anthropic", model=claude_model)
        assert len(response["content"]) > 0

def test_vertexai_claude_batch_convert_response(vertexai_claude_batch: VertexAIClaudeChatCompletionBatch):
    response = {
        "custom_id": "test_id",
        "status": "",
        "response": json.dumps({
            "id": "test_id",
            "model": claude_model,
            "stop_reason": "stop",
            "role": "assistant",
            "content": "Hello! How can I help?",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 8
            }
        })
    }
    
    converted = vertexai_claude_batch._convert_response(response)
    
    assert converted["id"] == "test_id"
    assert converted["custom_id"] == "test_id"
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == "Hello! How can I help?"
    assert converted["response"]["body"]["usage"]["prompt_tokens"] == 10
    assert converted["response"]["body"]["usage"]["completion_tokens"] == 8
    assert converted["response"]["body"]["usage"]["total_tokens"] == 18
    assert converted["error"] is None