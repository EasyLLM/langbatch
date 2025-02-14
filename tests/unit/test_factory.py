import os
import pytest
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
from langbatch.factory import chat_completion_batch, embedding_batch, get_args
from langbatch.errors import SetupError
from langbatch.anthropic import AnthropicChatCompletionBatch
from langbatch.openai import OpenAIChatCompletionBatch, OpenAIEmbeddingBatch
from langbatch.vertexai import VertexAIChatCompletionBatch
from langbatch.bedrock import BedrockClaudeChatCompletionBatch
from tests.unit.fixtures import test_data_file, temp_dir

def test_get_args():
    required_args = {
        "ENV_VAR": "arg_name",  # ENV_VAR is the env var name, arg_name is the kwarg name
        "OTHER_ENV": "other_arg",
        "THIRD_ENV": "third_arg"
    }
    
    # Test with all args in kwargs using the mapped names
    kwargs = {
        "arg_name": "value1",
        "other_arg": "value2",
        "third_arg": "value3"
    }
    extracted, missed = get_args(required_args, kwargs)
    assert extracted == {"arg_name": "value1", "other_arg": "value2", "third_arg": "value3"}
    assert len(missed) == 0

    # Test with env vars
    os.environ["ENV_VAR"] = "env_value1"
    kwargs = {
        "other_arg": "value2"  # Using mapped name in kwargs
    }
    extracted, missed = get_args(required_args, kwargs)
    assert extracted["arg_name"] == "env_value1"  # From env var
    assert extracted["other_arg"] == "value2"     # From kwargs
    assert "third_arg" in missed                  # Missing
    assert len(missed) == 1

    # Test with wrong key in kwargs (using env var name instead of mapped name)
    kwargs = {
        "ENV_VAR": "wrong_value",  # This should be ignored as it uses env var name
        "other_arg": "value2"      # This is correct as it uses mapped name
    }
    extracted, missed = get_args(required_args, kwargs)
    assert extracted["arg_name"] == "env_value1"  # Still from env var
    assert extracted["other_arg"] == "value2"     # From kwargs
    assert "third_arg" in missed
    assert len(missed) == 1

    # Test with all env vars
    os.environ["ENV_VAR"] = "env_value1"
    os.environ["OTHER_ENV"] = "env_value2"
    os.environ["THIRD_ENV"] = "env_value3"
    extracted, missed = get_args(required_args, {})
    assert extracted == {
        "arg_name": "env_value1",
        "other_arg": "env_value2",
        "third_arg": "env_value3"
    }
    assert len(missed) == 0

    # Cleanup
    del os.environ["ENV_VAR"]
    del os.environ["OTHER_ENV"]
    del os.environ["THIRD_ENV"]

def test_chat_completion_batch_anthropic(test_data_file, monkeypatch):
    # Test with API key in env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
    batch = chat_completion_batch(test_data_file, "anthropic")
    assert isinstance(batch, AnthropicChatCompletionBatch)
    assert batch._client is not None
    assert batch._client.api_key == "test_key"
    
    # Test with client in kwargs
    client = Anthropic(api_key="test_key")
    batch = chat_completion_batch(test_data_file, "anthropic", client=client)
    assert isinstance(batch, AnthropicChatCompletionBatch)
    assert batch._client is not None
    assert batch._client.api_key == "test_key"
    
    # Test missing API key
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SetupError, match="Anthropic API key not found"):
        chat_completion_batch(test_data_file, "anthropic")

def test_chat_completion_batch_openai(test_data_file, monkeypatch):
    # Test with API key in env
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    batch = chat_completion_batch(test_data_file, "openai")
    assert isinstance(batch, OpenAIChatCompletionBatch)
    assert batch._client is not None
    assert batch._client.api_key == "test_key"
    
    # Test with client in kwargs
    client = OpenAI(api_key="test_key")
    batch = chat_completion_batch(test_data_file, "openai", client=client)
    assert isinstance(batch, OpenAIChatCompletionBatch)
    assert batch._client is not None
    assert batch._client.api_key == "test_key"
    # Test missing API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SetupError, match="OpenAI API key not found"):
        chat_completion_batch(test_data_file, "openai")

def test_chat_completion_batch_azure(test_data_file, monkeypatch):
    # Test with all required args in env
    monkeypatch.setenv("AZURE_API_BASE", "test_endpoint")
    monkeypatch.setenv("AZURE_API_KEY", "test_key")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    batch = chat_completion_batch(test_data_file, "azure")
    assert isinstance(batch, OpenAIChatCompletionBatch)
    assert isinstance(batch._client, AzureOpenAI)
    assert batch._client is not None
    assert batch._client.api_key == "test_key"
    assert batch._client._api_version == "2024-10-21"

    # Test with args in kwargs
    batch = chat_completion_batch(
        test_data_file, 
        "azure",
        AZURE_API_BASE="test_endpoint",
        AZURE_API_KEY="test_key"
    )
    assert isinstance(batch, OpenAIChatCompletionBatch)
    assert isinstance(batch._client, AzureOpenAI)
    assert batch._client is not None
    assert batch._client.api_key == "test_key"
    assert batch._client._api_version == "2024-10-21"
    
    # Test missing required args
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    with pytest.raises(SetupError, match="Azure OpenAI requires the following"):
        chat_completion_batch(test_data_file, "azure")

def test_chat_completion_batch_vertex_ai(test_data_file, monkeypatch):
    required_args = {
        "gcp_project": "test-project",
        "bigquery_input_dataset": "test_input",
        "bigquery_output_dataset": "test_output"
    }
    model = "gemini-2.0-flash"
    
    # Test with env vars
    monkeypatch.setenv("GCP_PROJECT", required_args["gcp_project"])
    monkeypatch.setenv("GCP_BIGQUERY_INPUT_DATASET", required_args["bigquery_input_dataset"])
    monkeypatch.setenv("GCP_BIGQUERY_OUTPUT_DATASET", required_args["bigquery_output_dataset"])
    
    batch = chat_completion_batch(test_data_file, "vertex_ai", model=model)
    assert isinstance(batch, VertexAIChatCompletionBatch)
    assert batch.bigquery_input_dataset == required_args["bigquery_input_dataset"]
    assert batch.bigquery_output_dataset == required_args["bigquery_output_dataset"]
    assert batch.gcp_project == required_args["gcp_project"]
    assert batch.model == model

    # Test with args in kwargs
    batch = chat_completion_batch(
        test_data_file, 
        "vertex_ai",
        model=model,
        gcp_project=required_args["gcp_project"],
        bigquery_input_dataset=required_args["bigquery_input_dataset"],
        bigquery_output_dataset=required_args["bigquery_output_dataset"]
    )
    assert isinstance(batch, VertexAIChatCompletionBatch)
    assert batch.bigquery_input_dataset == required_args["bigquery_input_dataset"]
    assert batch.bigquery_output_dataset == required_args["bigquery_output_dataset"]
    assert batch.gcp_project == required_args["gcp_project"]
    assert batch.model == model
    
    # Test missing model
    with pytest.raises(SetupError, match="model is required for VertexAI"):
        chat_completion_batch(test_data_file, "vertex_ai")
    
    # Test invalid model
    with pytest.raises(SetupError, match="Invalid model for VertexAI"):
        chat_completion_batch(test_data_file, "vertex_ai", model="invalid-model")

def test_chat_completion_batch_bedrock(test_data_file, monkeypatch):
    required_args = {
        "input_bucket": "test-input",
        "output_bucket": "test-output",
        "region": "us-east-1",
        "service_role": "test-role"
    }
    model = "us.anthropic.claude-3"
    
    # Test with env vars
    monkeypatch.setenv("AWS_INPUT_BUCKET", required_args["input_bucket"])
    monkeypatch.setenv("AWS_OUTPUT_BUCKET", required_args["output_bucket"])
    monkeypatch.setenv("AWS_REGION", required_args["region"])
    monkeypatch.setenv("AWS_SERVICE_ROLE", required_args["service_role"])
    
    batch = chat_completion_batch(test_data_file, "bedrock", model=model)
    assert isinstance(batch, BedrockClaudeChatCompletionBatch)
    assert batch.input_bucket == required_args["input_bucket"]
    assert batch.output_bucket == required_args["output_bucket"]
    assert batch.region == required_args["region"]
    assert batch.service_role == required_args["service_role"]
    assert batch.model == model

    # Test with args in kwargs
    batch = chat_completion_batch(
        test_data_file, 
        "bedrock",
        model=model,
        input_bucket=required_args["input_bucket"],
        output_bucket=required_args["output_bucket"],
        region=required_args["region"],
        service_role=required_args["service_role"]
    )
    assert isinstance(batch, BedrockClaudeChatCompletionBatch)
    assert batch.input_bucket == required_args["input_bucket"]
    assert batch.output_bucket == required_args["output_bucket"]
    assert batch.region == required_args["region"]
    assert batch.service_role == required_args["service_role"]
    assert batch.model == model

    # Test missing model
    with pytest.raises(SetupError, match="model is required for Bedrock"):
        chat_completion_batch(test_data_file, "bedrock")
    
    # Test invalid model
    with pytest.raises(SetupError, match="Invalid model for Bedrock"):
        chat_completion_batch(test_data_file, "bedrock", model="invalid-model")

def test_chat_completion_batch_invalid_provider(test_data_file):
    with pytest.raises(SetupError, match="Invalid provider"):
        chat_completion_batch(test_data_file, "invalid_provider")

@pytest.mark.parametrize('test_data_file', ['embedding_batch.jsonl'], indirect=True)
def test_embedding_batch_openai(test_data_file, monkeypatch):
    # Test with API key in env
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    batch = embedding_batch(test_data_file, "openai")
    assert isinstance(batch, OpenAIEmbeddingBatch)
    
    # Test with client in kwargs
    client = OpenAI(api_key="test_key")
    batch = embedding_batch(test_data_file, "openai", client=client)
    assert isinstance(batch, OpenAIEmbeddingBatch)
    
    # Test missing API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SetupError, match="OpenAI API key not found"):
        embedding_batch(test_data_file, "openai")

@pytest.mark.parametrize('test_data_file', ['embedding_batch.jsonl'], indirect=True)
def test_embedding_batch_invalid_provider(test_data_file):
    with pytest.raises(SetupError, match="Invalid provider"):
        embedding_batch(test_data_file, "invalid_provider")
