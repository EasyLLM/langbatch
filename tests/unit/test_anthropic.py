from pathlib import Path
import time
import pytest
import jsonlines
import json
from langbatch.anthropic import AnthropicChatCompletionBatch
from langbatch.batch_storages import FileBatchStorage
from tests.unit.fixtures import test_data_file, temp_dir
from tests.unit.test_config import config

ANTHROPIC_COMPLETED_PLATFORM_BATCH_ID = config["anthropic"]["ANTHROPIC_COMPLETED_PLATFORM_BATCH_ID"]
model = config["anthropic"]["model"]

@pytest.fixture
def anthropic_batch(test_data_file: str):
    batch = AnthropicChatCompletionBatch(
        file=test_data_file)
    return batch

def test_anthropic_batch_init(anthropic_batch: AnthropicChatCompletionBatch):
    # check if the platform_batch_id is None
    assert anthropic_batch.platform_batch_id is None

    # check if the file is a file
    assert Path(anthropic_batch._file).is_file()

    # check if the file can be read as a jsonl file
    original_data = anthropic_batch._get_requests()
    for req in original_data:
        assert isinstance(req, dict)

def test_anthropic_batch_create(test_data_file):
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append(req)
    
    batch = AnthropicChatCompletionBatch.create_from_requests(requests)
    assert isinstance(batch, AnthropicChatCompletionBatch)
    
    # Check if the file is created and contains the correct number of requests
    batch_file_requests = []
    with jsonlines.open(batch._file) as reader:
        for req in reader:
            batch_file_requests.append(req)
    assert len(batch_file_requests) == len(requests)
    for req, batch_req in zip(requests, batch_file_requests):
        assert req == batch_req

def test_anthropic_batch_save_and_load(anthropic_batch: AnthropicChatCompletionBatch, temp_dir):
    storage = FileBatchStorage(temp_dir)
    
    # Save the batch
    anthropic_batch.save(storage=storage)
    
    # Load the batch
    loaded_batch = AnthropicChatCompletionBatch.load(anthropic_batch.id, storage=storage)
    
    assert loaded_batch.id == anthropic_batch.id    
    assert loaded_batch.platform_batch_id == anthropic_batch.platform_batch_id

    anthropic_batch.platform_batch_id = "new_platform_batch_id"
    anthropic_batch.save(storage=storage)
    loaded_batch = AnthropicChatCompletionBatch.load(anthropic_batch.id, storage=storage)
    assert loaded_batch.platform_batch_id == anthropic_batch.platform_batch_id

def test_anthropic_batch_get_status(anthropic_batch: AnthropicChatCompletionBatch, monkeypatch):
    with pytest.raises(ValueError, match="Batch not started"):
        anthropic_batch.get_status()

    anthropic_batch.platform_batch_id = ANTHROPIC_COMPLETED_PLATFORM_BATCH_ID
    assert anthropic_batch.get_status() == "completed"
    
    anthropic_batch.platform_batch_id = "platform_batch_id-1"
    monkeypatch.setattr(anthropic_batch, 'get_status', lambda: "in_progress")
    assert anthropic_batch.get_status() == "in_progress"

def test_anthropic_batch_convert_request(anthropic_batch: AnthropicChatCompletionBatch):
    request = {
        "custom_id": "test_id",
        "body": {
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
    }
    
    converted = anthropic_batch._convert_request(request)
    
    assert converted["custom_id"] == "test_id"
    assert converted["params"]["model"] == model
    assert converted["params"]["messages"][0]["role"] == "user"
    assert converted["params"]["messages"][0]["content"][0]["type"] == "text"
    assert converted["params"]["messages"][0]["content"][0]["text"] == "Hello, how are you?"
    assert converted["params"]["system"] == "You are a helpful assistant."
    assert converted["params"]["temperature"] == 0.7
    assert converted["params"]["max_tokens"] == 100
    assert len(converted["params"]["tools"]) == 1
    assert converted["params"]["tools"][0]["name"] == "get_weather"
    assert converted["params"]["tool_choice"] == {"type": "auto"}

def test_anthropic_batch_convert_response(anthropic_batch: AnthropicChatCompletionBatch):
    content = [
        {
            "type": "text",
            "text": "Hello again! It's nice to see you."
        }
    ]
    response = {
        "custom_id": "test_id",
        "result": {
            "type": "succeeded",
            "message": {
                "id":"msg_014VwiXbi91y3JMjcpyGBHX5",
                "type":"message",
                "role":"assistant",
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 36}
            }
        }
    }
    response["result"]["message"]["content"] = content

    # class CustomResponse(BaseModel):
    #     custom_id: str
    #     result: BetaMessageBatchSucceededResult
    # custom_response = CustomResponse(**response)
    
    converted = anthropic_batch._convert_response(response)
    
    assert converted["id"] == "test_id"
    assert converted["custom_id"] == "test_id"
    assert converted["response"]["body"]["choices"][0]["message"]["role"] == "assistant"
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == content[0]
    assert converted["response"]["body"]["usage"]["prompt_tokens"] == 11
    assert converted["response"]["body"]["usage"]["completion_tokens"] == 36
    assert converted["response"]["body"]["usage"]["total_tokens"] == 47
    assert converted["response"]["body"]["model"] == model
    assert converted["error"] is None

    content = [
        {
            "type": "text",
            "text": "<thinking>To answer this question, I will: 1. Use the get_weather tool</thinking>"
        },
        {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "get_weather",
            "input": {"location": "San Francisco, CA"}
        }
    ]
    response["result"]["message"]["content"] = content
    # custom_response = CustomResponse(**response)
    converted = anthropic_batch._convert_response(response)
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == content[0]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["id"] == content[1]["id"]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == content[1]["name"]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == json.dumps(content[1]["input"])

@pytest.mark.slow
def test_anthropic_batch_start(anthropic_batch: AnthropicChatCompletionBatch):
    requests = anthropic_batch._get_requests()

    converted_requests = []
    for req in requests:
        new_req = req.copy()
        new_req["body"]["model"] = model
        converted_requests.append(new_req)

    with jsonlines.open(anthropic_batch._file, mode="w") as writer:
        writer.write_all(converted_requests)

    anthropic_batch.start()
    assert anthropic_batch.platform_batch_id is not None

    time.sleep(5)
    assert anthropic_batch.get_status() == "in_progress"

def test_anthropic_batch_get_results(anthropic_batch: AnthropicChatCompletionBatch, monkeypatch):
    monkeypatch.setattr(anthropic_batch, 'platform_batch_id', ANTHROPIC_COMPLETED_PLATFORM_BATCH_ID)
    successful_results, unsuccessful_results = anthropic_batch.get_results()
    
    assert len(successful_results) > 0
    assert len(unsuccessful_results) == 0

    for successful_result in successful_results:
        assert successful_result["custom_id"] is not None
        assert successful_result["choices"] is not None
        assert len(successful_result["choices"]) > 0
        assert successful_result["choices"][0]["message"]["content"] is not None
