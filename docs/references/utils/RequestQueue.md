# RequestQueue Classes

::: langbatch.request_queues.RequestQueue
    options:
        show_root_toc_entry: false

!!!info
    Please make sure to pass the the correct type of requests to the queue as per the type of Batch you are going to use. For Example, if you are using ChatCompletionBatch as your batch type, then you should pass the requests in the same format as the  [ChatCompletionBatch.create()](../ChatCompletion.md/#langbatch.ChatCompletionBatch.ChatCompletionBatch.create) expects. For EmbeddingBatch, refer [EmbeddingBatch.create()](../Embedding.md/#langbatch.EmbeddingBatch.EmbeddingBatch.create)

::: langbatch.request_queues.InMemoryRequestQueue
    options:
        show_root_toc_entry: false
        inherit_members: true
        members:
            - add_requests
            - get_requests

::: langbatch.request_queues.RedisRequestQueue
    options:
        show_root_toc_entry: false
        inherit_members: true
        members:
            - __init__
            - add_requests
            - get_requests