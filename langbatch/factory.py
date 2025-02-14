import os

from anthropic import Anthropic
from openai import OpenAI, AzureOpenAI

from langbatch.errors import SetupError
from langbatch.anthropic import AnthropicChatCompletionBatch
from langbatch.bedrock import BedrockClaudeChatCompletionBatch
from langbatch.bedrock import BedrockNovaChatCompletionBatch
from langbatch.openai import OpenAIChatCompletionBatch, OpenAIEmbeddingBatch
from langbatch.vertexai import VertexAIChatCompletionBatch
from langbatch.vertexai import VertexAIClaudeChatCompletionBatch
from langbatch.vertexai import VertexAILlamaChatCompletionBatch

def get_args(required_args: dict, kwargs: dict):
    extracted_args = {}
    missed_args = []
    for arg in required_args.keys():
        if required_args[arg] in kwargs:
            extracted_args[required_args[arg]] = kwargs[required_args[arg]]
        else:
            env_arg =os.getenv(arg)
            if env_arg:
                extracted_args[required_args[arg]] = env_arg
            else:
                missed_args.append(required_args[arg])

    return extracted_args, missed_args

def chat_completion_batch(file: str, provider: str, model: str = None, **kwargs):
    if provider == "anthropic":
        if len(kwargs) == 0 or "client" not in kwargs:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                anthropic_client = Anthropic(api_key=api_key)
                return AnthropicChatCompletionBatch(file, anthropic_client)
            else:
                raise SetupError("Anthropic API key not found")
        else:
            return AnthropicChatCompletionBatch(file, **kwargs)
    elif provider == "openai":
        if len(kwargs) == 0 or "client" not in kwargs:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai_client = OpenAI(api_key=api_key)
                return OpenAIChatCompletionBatch(file, openai_client)
            else:
                raise SetupError("OpenAI API key not found")
        else:
            return OpenAIChatCompletionBatch(file, **kwargs)
    elif provider == "azure":
        required_args = {
            "AZURE_API_BASE":"azure_endpoint",
            "AZURE_API_KEY":"api_key",
            "AZURE_API_VERSION":"api_version",
        }
        extracted_args, missed_args = get_args(required_args, kwargs)
        if len(missed_args) > 0:
            if "api_version" in missed_args:
                extracted_args["api_version"] = "2025-01-01-preview"
                missed_args.remove("api_version")
            if "api_key" in missed_args and "azure_ad_token_provider" in kwargs:
                extracted_args["azure_ad_token_provider"] = kwargs["azure_ad_token_provider"]
                missed_args.remove("api_key")

        if len(missed_args) > 0:
            raise SetupError(f"Azure OpenAI requires the following: {missed_args}")
        else:
            azure_client = AzureOpenAI(**extracted_args)
            return OpenAIChatCompletionBatch(file, azure_client)
    elif provider == "vertex_ai":
        required_args = {
            "GCP_PROJECT":"gcp_project", 
            "GCP_BIGQUERY_INPUT_DATASET":"bigquery_input_dataset", 
            "GCP_BIGQUERY_OUTPUT_DATASET":"bigquery_output_dataset"
        }
        extracted_args, missed_args = get_args(required_args, kwargs)
        if len(missed_args) > 0:
            raise SetupError(f"VertexAI requires the following: {missed_args}")
        else:
            if model:
                if model.startswith("gemini"):
                    return VertexAIChatCompletionBatch(file, model, **extracted_args)
                elif model.startswith("claude"):
                    return VertexAIClaudeChatCompletionBatch(file, model, **extracted_args)
                elif model.startswith("llama"):
                    return VertexAILlamaChatCompletionBatch(file, model, **extracted_args)
                else:
                    raise SetupError(f"Invalid model for VertexAI: {model}")
            else:
                raise SetupError("model is required for VertexAI")
    elif provider == "bedrock":
        required_args = {
            "AWS_INPUT_BUCKET":"input_bucket",
            "AWS_OUTPUT_BUCKET":"output_bucket",
            "AWS_REGION":"region",
            "AWS_SERVICE_ROLE":"service_role"
        }
        extracted_args, missed_args = get_args(required_args, kwargs)
        if len(missed_args) > 0:
            raise SetupError(f"Bedrock requires the following: {missed_args}")
        else:
            if model:
                if model.startswith("us.anthropic.claude"):
                    return BedrockClaudeChatCompletionBatch(file, model, **extracted_args)
                elif model.startswith("us.amazon.nova"):
                    return BedrockNovaChatCompletionBatch(file, model, **extracted_args)
                else:
                    raise SetupError(f"Invalid model for Bedrock: {model}")
            else:
                raise SetupError("model is required for Bedrock")
    else:
        raise SetupError(f"Invalid provider: {provider}")
    
def embedding_batch(file: str, provider: str, model: str = None, **kwargs):
    if provider == "openai":
        if len(kwargs) == 0 or "client" not in kwargs:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai_client = OpenAI(api_key=api_key)
                return OpenAIEmbeddingBatch(file, openai_client)
            else:
                raise SetupError("OpenAI API key not found")
        else:
            return OpenAIEmbeddingBatch(file, **kwargs)
    else:
        raise SetupError(f"Invalid provider: {provider}")