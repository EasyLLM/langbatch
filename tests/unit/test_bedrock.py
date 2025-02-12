# Test cases for Bedrock

import os
from pathlib import Path
import json
import time
import pytest
import jsonlines
from langbatch.bedrock import BedrockClaudeChatCompletionBatch, BedrockNovaChatCompletionBatch
from langbatch.batch_storages import FileBatchStorage
from tests.unit.fixtures import test_data_file, temp_dir

BEDROCK_COMPLETED_PLATFORM_BATCH_ID = os.environ.get('BEDROCK_COMPLETED_PLATFORM_BATCH_ID')
BEDROCK_COMPLETED_BATCH_ID = os.environ.get('BEDROCK_COMPLETED_BATCH_ID')

service_role = os.environ.get('BEDROCK_SERVICE_ROLE')
claude_model_name = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
nova_model_name = "us.amazon.nova-lite-v1:0"

input_bucket_nova = os.environ.get('S3_INPUT_BUCKET_NOVA')
output_bucket_nova = os.environ.get('S3_OUTPUT_BUCKET_NOVA')
region_nova = os.environ.get('AWS_REGION_NOVA')

input_bucket_claude = os.environ.get('S3_INPUT_BUCKET_CLAUDE')
output_bucket_claude = os.environ.get('S3_OUTPUT_BUCKET_CLAUDE')
region_claude = os.environ.get('AWS_REGION_CLAUDE')

@pytest.fixture
def bedrock_claude_batch(test_data_file: str):
    batch = BedrockClaudeChatCompletionBatch(
        file=test_data_file,
        model_name=claude_model_name,
        input_bucket=input_bucket_claude,
        output_bucket=output_bucket_claude,
        region=region_claude,
        service_role=service_role
    )
    return batch

@pytest.fixture
def bedrock_nova_batch(test_data_file: str):
    batch = BedrockNovaChatCompletionBatch(
        file=test_data_file,
        model_name=nova_model_name,
        input_bucket=input_bucket_nova,
        output_bucket=output_bucket_nova,
        region=region_nova,
        service_role=service_role
    )
    return batch

def test_bedrock_batch_init(bedrock_claude_batch: BedrockClaudeChatCompletionBatch):
    # check if the platform_batch_id is None
    assert bedrock_claude_batch.platform_batch_id is None
    assert bedrock_claude_batch.model_name == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert bedrock_claude_batch.input_bucket == input_bucket_claude
    assert bedrock_claude_batch.output_bucket == output_bucket_claude
    assert bedrock_claude_batch.region == region_claude
    assert bedrock_claude_batch.service_role == service_role

    # check if the file is a file
    assert Path(bedrock_claude_batch._file).is_file()

    # check if the file can be read as a jsonl file
    original_data = bedrock_claude_batch._get_requests()
    for req in original_data:
        assert isinstance(req, dict)

def test_bedrock_batch_create(test_data_file):
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append(req)
    
    batch = BedrockClaudeChatCompletionBatch.create_from_requests(
        requests,
        batch_kwargs={
            "model_name": claude_model_name,
            "input_bucket": input_bucket_claude,
            "output_bucket": output_bucket_claude,
            "region": region_claude,
            "service_role": service_role
        }
    )
    assert isinstance(batch, BedrockClaudeChatCompletionBatch)
    assert batch.model_name == claude_model_name
    assert batch.input_bucket == input_bucket_claude
    assert batch.output_bucket == output_bucket_claude
    assert batch.region == region_claude
    assert batch.service_role == service_role
    
    # Check if the file is created and contains the correct number of requests
    batch_file_requests = []
    with jsonlines.open(batch._file) as reader:
        for req in reader:
            batch_file_requests.append(req)
    assert len(batch_file_requests) == len(requests)
    for req, batch_req in zip(requests, batch_file_requests):
        assert req == batch_req

def test_bedrock_batch_save_and_load(bedrock_claude_batch: BedrockClaudeChatCompletionBatch, temp_dir):
    storage = FileBatchStorage(temp_dir)
    
    # Save the batch
    bedrock_claude_batch.save(storage=storage)
    
    # Load the batch
    loaded_batch = BedrockClaudeChatCompletionBatch.load(bedrock_claude_batch.id, storage=storage)
    
    assert loaded_batch.id == bedrock_claude_batch.id    
    assert loaded_batch.platform_batch_id == bedrock_claude_batch.platform_batch_id
    assert loaded_batch.model_name == bedrock_claude_batch.model_name
    assert loaded_batch.input_bucket == bedrock_claude_batch.input_bucket
    assert loaded_batch.output_bucket == bedrock_claude_batch.output_bucket
    assert loaded_batch.region == bedrock_claude_batch.region
    assert loaded_batch.service_role == bedrock_claude_batch.service_role

    bedrock_claude_batch.platform_batch_id = "new_platform_batch_id"
    bedrock_claude_batch.save(storage=storage)
    loaded_batch = BedrockClaudeChatCompletionBatch.load(bedrock_claude_batch.id, storage=storage)
    assert loaded_batch.platform_batch_id == bedrock_claude_batch.platform_batch_id

def test_bedrock_batch_get_status(bedrock_claude_batch: BedrockClaudeChatCompletionBatch, monkeypatch):
    with pytest.raises(ValueError, match="Batch not started"):
        bedrock_claude_batch.get_status()

    bedrock_claude_batch.platform_batch_id = BEDROCK_COMPLETED_PLATFORM_BATCH_ID
    monkeypatch.setattr(bedrock_claude_batch, 'id', BEDROCK_COMPLETED_BATCH_ID)
    assert bedrock_claude_batch.get_status() == "completed"
    
    bedrock_claude_batch.platform_batch_id = "platform_batch_id-1"
    monkeypatch.setattr(bedrock_claude_batch, 'get_status', lambda: "in_progress")
    assert bedrock_claude_batch.get_status() == "in_progress"

def test_bedrock_claude_batch_convert_request(bedrock_claude_batch: BedrockClaudeChatCompletionBatch):
    request = {
        "custom_id": "test_id",
        "body": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "model": claude_model_name,
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
    
    converted = bedrock_claude_batch._convert_request(request)
    
    assert converted["recordId"] == "test_id"
    assert "model" not in converted["modelInput"]
    assert "anthropic_version" in converted["modelInput"]
    assert converted["modelInput"]["messages"][0]["role"] == "user"
    assert converted["modelInput"]["messages"][0]["content"][0]["type"] == "text"
    assert converted["modelInput"]["messages"][0]["content"][0]["text"] == "Hello, how are you?"
    assert converted["modelInput"]["system"] == "You are a helpful assistant."
    assert converted["modelInput"]["temperature"] == 0.7
    assert converted["modelInput"]["max_tokens"] == 100
    assert len(converted["modelInput"]["tools"]) == 1
    assert converted["modelInput"]["tools"][0]["name"] == "get_weather"

def test_bedrock_claude_batch_convert_response(bedrock_claude_batch: BedrockClaudeChatCompletionBatch):
    content = [
        {
            "type": "text",
            "text": "Hello again! It's nice to see you."
        }
    ]
    response = {
        "recordId": "test_id",
        "modelOutput": {
            "id":"msg_014VwiXbi91y3JMjcpyGBHX5",
            "type":"message",
            "role":"assistant",
            "model": claude_model_name,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 36},
            "content": content
        }
    }
    
    converted = bedrock_claude_batch._convert_response(response)
    
    assert converted["id"] == "test_id"
    assert converted["custom_id"] == "test_id"
    assert converted["response"]["body"]["choices"][0]["message"]["role"] == "assistant"
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == content[0]
    assert converted["response"]["body"]["usage"]["prompt_tokens"] == 11
    assert converted["response"]["body"]["usage"]["completion_tokens"] == 36
    assert converted["response"]["body"]["usage"]["total_tokens"] == 47
    assert converted["response"]["body"]["model"] == claude_model_name
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
    response["modelOutput"]["content"] = content
    converted = bedrock_claude_batch._convert_response(response)
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == content[0]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["id"] == content[1]["id"]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == content[1]["name"]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == content[1]["input"]

def test_bedrock_nova_batch_convert_request(bedrock_nova_batch: BedrockNovaChatCompletionBatch):
    request = {
        "custom_id": "test_id",
        "body": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "model": nova_model_name,
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
    
    converted = bedrock_nova_batch._convert_request(request)
    
    assert converted["recordId"] == "test_id"
    assert "model" not in converted["modelInput"]
    assert "messages" in converted["modelInput"]
    assert "temperature" in converted["modelInput"]["inferenceConfig"]
    assert "max_new_tokens" in converted["modelInput"]["inferenceConfig"]
    assert "toolConfig" in converted["modelInput"]

def test_bedrock_nova_batch_convert_response(bedrock_nova_batch: BedrockNovaChatCompletionBatch):
    response = {
        "recordId": "test_id",
        "modelOutput": {
            "output": {
                "message": {
                    "content": [
                    {
                        "text": "Hello! How can I help you today?"
                    }
                    ],
                    "role": "assistant"
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 8,
                "totalTokens": 18
            }
        }
    }

    converted = bedrock_nova_batch._convert_response(response)
    
    assert converted["id"] == "test_id"
    assert converted["custom_id"] == "test_id"
    assert converted["response"]["body"]["choices"][0]["message"]["role"] == "assistant"
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
    assert converted["response"]["body"]["usage"]["prompt_tokens"] == 10
    assert converted["response"]["body"]["usage"]["completion_tokens"] == 8
    assert converted["response"]["body"]["usage"]["total_tokens"] == 18
    assert converted["response"]["body"]["model"] == nova_model_name
    assert converted["error"] is None

    response = {
        "recordId": "test_id",
        "modelOutput": {
            "output": {
                "message": {
                    "content": [
                    {
                        "text": "<thinking>I need to use the 'get_weather' tool to find out the weather in Bangalore. I will call this tool with the appropriate argument.</thinking>\n"
                    },
                    {
                        "toolUse": {
                        "name": "get_weather",
                        "toolUseId": "d09f2b7d-9fdc-4bad-bb65-13f310b8d002",
                        "input": {
                            "city": "Bangalore"
                        }
                        }
                    }
                    ],
                    "role": "assistant"
                }
            },
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 432,
                "outputTokens": 88,
                "totalTokens": 520
            }
        }
    }

    converted = bedrock_nova_batch._convert_response(response)
    assert converted["response"]["body"]["choices"][0]["message"]["content"] == response["modelOutput"]["output"]["message"]["content"][0]["text"]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["id"] == response["modelOutput"]["output"]["message"]["content"][1]["toolUse"]["toolUseId"]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == response["modelOutput"]["output"]["message"]["content"][1]["toolUse"]["name"]
    assert converted["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == json.dumps(response["modelOutput"]["output"]["message"]["content"][1]["toolUse"]["input"])

@pytest.mark.slow
def test_bedrock_batch_start(bedrock_claude_batch: BedrockClaudeChatCompletionBatch):
    bedrock_claude_batch.start()
    assert bedrock_claude_batch.platform_batch_id is not None

    time.sleep(5)
    assert bedrock_claude_batch.get_status() == "in_progress"

def test_bedrock_batch_get_results(bedrock_claude_batch: BedrockClaudeChatCompletionBatch, monkeypatch):
    monkeypatch.setattr(bedrock_claude_batch, 'platform_batch_id', BEDROCK_COMPLETED_PLATFORM_BATCH_ID)
    monkeypatch.setattr(bedrock_claude_batch, 'id', BEDROCK_COMPLETED_BATCH_ID)
    
    successful_results, unsuccessful_results = bedrock_claude_batch.get_results()
    
    assert len(successful_results) > 0
    assert len(unsuccessful_results) == 0

    for successful_result in successful_results:
        assert successful_result["custom_id"] is not None
        assert successful_result["choices"] is not None
        assert len(successful_result["choices"]) > 0
        assert successful_result["choices"][0]["message"]["content"] is not None
