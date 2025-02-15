<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="./docs/_static/imgs/langbatch-logo.png">
</h1>
<p align="center">
  <i>Call all Batch APIs using the OpenAI format [OpenAI, Anthropic, Azure OpenAI, Vertex AI, Bedrock]</i>
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
    <a href="https://discord.gg/uh7PZ2XT">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/uh7PZ2XT?style=flat">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://www.langbatch.com/">Documentation</a> |
        <a href="https://discord.gg/uh7PZ2XT">Join Discord</a> 
    <p>
</h4>

LangBatch is a unified API for accessing Batch APIs from providers like OpenAI, Anthropic, Azure OpenAI, Vertex AI, Bedrock, etc.  

## Utlize Batch APIs for

- Request that don't require immediate responses.
- Low cost (usually 50% discount)
- Higher rate limits
- Example use cases: Data processing pipelines, Evaluations, Classifying huge data, Creating embeddings for large text contents

## Key Features

- OpenAI format for requests and responses
- Utilities for handling the complete lifecycle of a batch job: Creating, Starting, Monitoring, Retrying and Processing
- Pipeline to convert stream of incoming requests into batch jobs

## Installation

PyPI: 

```bash
pip install langbatch
```

Alternatively, from source:

```bash
pip install git+https://github.com/EasyLLM/langbatch
```

Find the complete [Installation](https://www.langbatch.com/installation/) guide here.

### Quickstart

Here is the 3 main lines to start a batch job:
```python
from langbatch import chat_completion_batch
batch = chat_completion_batch("openai_chat_completion_requests.jsonl", provider="openai")
batch.start()
```

Check the status of the batch and get the results:
```python
if batch.get_status() == "completed":
    results, _ = batch.get_results()
    for result in results:
        print(f"Custom ID: {result['custom_id']}")
        print(f"Content: {result['choices'][0]['message']['content']}")
```

Find the complete [Get Started](https://www.langbatch.com/getstarted/batch/) guide here.

## 🫂 Community

If you want to get more involved with LangBatch, check out our [discord server](https://discord.gg/uh7PZ2XT)

## Contributors

```yml
+----------------------------------------------------------------------------+
|     +----------------------------------------------------------------+     |
|     | Developers: Those who built with `langbatch`.                  |     |
|     | (You have `import langbatch` somewhere in your project)        |     |
|     |     +----------------------------------------------------+     |     |
|     |     | Contributors: Those who make `langbatch` better.   |     |     |
|     |     | (You make PR to this repo)                         |     |     |
|     |     +----------------------------------------------------+     |     |
|     +----------------------------------------------------------------+     |
+----------------------------------------------------------------------------+
```

We welcome contributions from the community! Whether it's bug fixes, feature additions, or documentation improvements, your input is valuable.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request