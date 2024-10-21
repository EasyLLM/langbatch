# Embedding Batch

`EmbeddingBatch` is the abstract batch class for processing embedding requests. It utilizes the OpenAI Embedding API format as its standard request structure.

1. **Universal Compatibility**: While based on the OpenAI format, `EmbeddingBatch` can be used with embedding models from various providers, not just OpenAI.

2. **Extensibility**: Serves as a base class for creating specialized embedding batch classes.

3. **Standardized Input/Output**: Uses OpenAI Batch File format and the OpenAI Embedding API format for input. Results are also consistently provided in the OpenAI format.

By standardizing on the OpenAI format, `EmbeddingBatch` ensures consistency and interoperability across different embedding implementations within the LangBatch ecosystem.

## Initialize an Embedding Batch

You can initialize an `EmbeddingBatch` by passing the path to a file. The file should be in OpenAI Batch File format.

```python
from langbatch import OpenAIEmbeddingBatch

batch = OpenAIEmbeddingBatch("data.jsonl")
```

You can also pass a list of texts to the batch.

```python
batch = OpenAIEmbeddingBatch.create([
    "Lincoln was the 16th President of the United States. His face is on Mount Rushmore.", 
    "Steve Jobs was the co-founder of Apple Inc. He was considered a visionary and a pioneer in the personal computer revolution."
])

# Initializing with request kwargs
batch = OpenAIEmbeddingBatch.create([
    "Hello World",
    "Hello LangBatch"
], request_kwargs={"model": "text-embedding-3-large"})
```

## Get Results

```python
successful_results, unsuccessful_results = batch.get_results()
for result in successful_results:
    print(result["embedding"])
```