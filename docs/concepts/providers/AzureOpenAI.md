# Azure OpenAI

You can run batch inference jobs on Azure OpenAI models via LangBatch.

## Data Format
Our default OpenAI data format can be used for Azure OpenAI. But you need to replace the model name with the model deployment name.

???+ note
    You need to choose 'Global Batch' for deployment type when creating the model deployment in Azure OpenAI. And by setting the deployment name as the model name (For example, `gpt-4o` or `gpt-4o-mini`), you can use the same OpenAI model names in the batch file.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

## Create Chat Completion Batch

```python
import os
from langbatch import chat_completion_batch
os.environ["AZURE_API_KEY"] = "your-azure-openai-api-key"
os.environ["AZURE_API_BASE"] = "https://{resource-name}.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-02-15-preview"

batch = chat_completion_batch("path/to/batch-file.jsonl", provider="azure")
```

You can also pass the configuration values as arguments.

```python
batch = chat_completion_batch(
    "path/to/batch-file.jsonl", 
    provider="azure", 
    api_key="your-azure-openai-api-key",
    azure_endpoint="https://{resource-name}.openai.azure.com/", 
    api_version="2024-02-15-preview"
)
```

You can pass 'azure_ad_token_provider' instead of 'api_key'.

```python
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

batch = chat_completion_batch(
    "path/to/batch-file.jsonl", 
    provider="azure", 
    azure_ad_token_provider=token_provider,
    azure_endpoint="https://{resource-name}.openai.azure.com/", 
    api_version="2024-02-15-preview"
)
```

!!! info
    Azure OpenAI does not support Text Embedding models and Finetuned LLMs yet.
    Refer to [Azure OpenAI Batch API Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?pivots=programming-language-python){:target="_blank"} for more information.
