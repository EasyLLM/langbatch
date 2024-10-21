# Installation

To get started, install LangBatch using `pip` with the following command:

```bash
pip install langbatch
```

This will install the core LangBatch package. 

## Optional Provider Dependencies
Core LangBatch package only supports OpenAI and OpenAI compatible providers. For Other providers, you need to install additional dependencies.

- Anthropic:
```bash
pip install langbatch[Anthropic]
```

This will install the dependencies for using the Anthropic with LangBatch.

- Vertex AI:
```bash
pip install langbatch[VertexAI]
```

This will install the dependencies for using the Vertex AI with LangBatch.

## Utility and Integration Dependencies
These are optional dependencies for using other utilities and integrations.

- Redis:
```bash
pip install langbatch[redis]
```

This will install the dependencies for using RedisRequestQueue with LangBatch.

## Install all dependencies
```bash
pip install langbatch[all]
```

This will install the dependencies for using all the utilities and integrations with LangBatch.

## Install from main branch
If you'd like to experiment with the latest features, install the most recent version from the main branch:

```bash
pip install git+https://github.com/EasyLLM/langbatch.git
```

## Install from source
If you're planning to contribute and make modifications to the code, ensure that you clone the repository and set it up as an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs){:target="_blank"}.

```bash
git clone https://github.com/EasyLLM/langbatch.git 
cd langbatch 
pip install -e .
```

This will enable you to run LangBatch locally from the source code with immediate effect on changes without needing to reinstall.

Next, let's create a OpenAIChatCompletionBatch object and perform batch generations.