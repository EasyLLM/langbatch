# Azure OpenAI

You can utilize Azure OpenAI Batch API for running batch generations on Azure OpenAI models via LangBatch.

## Data Format
Our default OpenAI data format can be used for Azure OpenAI. But you need to replace the model name with the model deployment name.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"model": "REPLACE-WITH-MODEL-DEPLOYMENT-NAME", "messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

## Azure OpenAI Client

Azure OpenAI client is used to make requests to the Azure OpenAI service.

```python
from openai import AzureOpenAI
from langbatch.openai import OpenAIChatCompletionBatch
    
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-07-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

batch = OpenAIChatCompletionBatch(file="data.jsonl", client=client)

batch.start()
```

!!! info
    Azure OpenAI does not support Text Embedding models and Finetuned LLMs yet.
    Refer to [Azure OpenAI Batch APIDocumentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?pivots=programming-language-python){:target="_blank"} for more information.
