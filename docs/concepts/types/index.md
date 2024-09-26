# Batch and Batch Types 
Batch is the core building block of LangBatch. A batch instance is created with a JSONL file containing lot of individual requests. 
This collection of requests is sent to an AI provider as a single unit when we start a batch. 

There are different types of batches in LangBatch for different types of AI tasks. For Example, LangBatch has `ChatCompletionBatch` for Chat Completion tasks with Large Language Models and `EmbeddingBatch` for creating text embeddings. We will add more batch types in future to support more AI tasks.

<div class="grid cards" markdown>

- [Batch Class](./batch.md)

    How to create batch?

- [Chat Completion](./chat_completion.md)

    How to create a batch for Chat Completion?

- [Embedding](./embedding.md)

    How to create a batch for Embedding?

</div>
