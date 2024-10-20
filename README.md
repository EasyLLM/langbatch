<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="./docs/_static/imgs/langbatch-logo.png">
</h1>
<p align="center">
  <i>Call all Batch APIs using the OpenAI format [OpenAI, Anthropic, Azure OpenAI, VertexAI]</i>
</p>

<p align="center">
    <a href="https://github.com/EasyLLM/langbatch/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/EasyLLM/langbatch.svg">
    </a>
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/EasyLLM/langbatch/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/EasyLLM/langbatch.svg?color=green">
    </a>
    <a href="https://pypi.org/project/langbatch/">
        <img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/langbatch">
    </a>
    <a href="https://discord.gg/7FS87Rfb">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/7FS87Rfb?style=flat">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://langbatch.easyllm.tech/">Documentation</a> |
        <a href="https://discord.gg/7FS87Rfb">Join Discord</a> 
    <p>
</h4>

LangBatch is a Python library for large scale AI generation using batch APIs from providers like OpenAI, Azure OpenAI, GCP VertexAI, etc.  

## Utlize Batch APIs for

- Request that don't require immediate responses.
- Low cost (usually 50% discount)
- Higher rate limits
- Example use cases: Data processing pipelines, Evaluations, Classifying huge data, Creating embeddings for large text contents

## Key Features

- Unified API to access Batch APIs from different providers.
- Standarized OpenAI format for requests and responses
- Utilities for handling the complete lifecycle of a batch job: Creating, Starting, Monitoring, Retrying and Processing Completed

## Installation

PyPI: 

```bash
pip install langbatch
```

Alternatively, from source:

```bash
pip install git+https://github.com/EasyLLM/langbatch
```

### Quickstart

Here is the 3 main lines to start a batch job:
```python
from langbatch import OpenAIChatCompletionBatch
batch = OpenAIChatCompletionBatch("openai_chat_completion_requests.jsonl")
batch.start()
```

Find the complete [Get Started](https://langbatch.easyllm.tech/getstarted/batch/) guide here.