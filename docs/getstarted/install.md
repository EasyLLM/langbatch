# Installation

To get started, install LangBatch using `pip` with the following command:

```bash
pip install langbatch
```

If you'd like to experiment with the latest features, install the most recent version from the main branch:

```bash
pip install git+https://github.com/EasyLLM/langbatch.git
```

If you're planning to contribute and make modifications to the code, ensure that you clone the repository and set it up as an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).

```bash
git clone https://github.com/EasyLLM/langbatch.git 
cd langbatch 
pip install -e .
```

Next, let's create a OpenAIChatCompletionBatch object and perform batch generations.