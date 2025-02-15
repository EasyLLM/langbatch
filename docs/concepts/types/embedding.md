# Embedding Batch

`EmbeddingBatch` is the abstract batch class for processing embedding requests. Same as `ChatCompletionBatch`, it uses the OpenAI Batch File format for requests and responses.

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

In EmbeddingBatch, the successful results contain `embedding` and `custom_id` keys.

```python
successful_results, unsuccessful_results = batch.get_results()
for result in successful_results:
    print(f"Custom ID: {result['custom_id']}")
    print(result["embedding"])
```