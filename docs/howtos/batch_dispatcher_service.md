# A sample API service - Stream to Batch pipeline

In production, you may want to implement a REST API service that accepts invidual requests or list of requests and process them as batches later. 

Refer to [Batch Dispatcher](../concepts/pipeline/batch_dispatcher.md) for more details.


```python
# main.py
import logging
import asyncio
from typing import List
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from langbatch import BatchHandler
from langbatch.batch_queues import FileBatchQueue
from langbatch.batch_storages import FileBatchStorage
from langbatch.openai import OpenAIChatCompletionBatch
from langbatch import BatchDispatcher
from langbatch.request_queues import InMemoryRequestQueue

logging.basicConfig(level=logging.INFO)

# Function to process the successfully completed batch
def process_batch(batch: OpenAIChatCompletionBatch):
    successful_results, unsuccessful_results = batch.get_results()
    for successful_result in successful_results:
        print(successful_result["custom_id"])
        print(successful_result["choices"][0]["message"]["content"])

        # TODO: process the successful result

# Initialize Batch Handler and Batch Dispatcher
batch_queue = FileBatchQueue("batch_queue.json")
batch_storage = FileBatchStorage()
handler = BatchHandler(
    batch_process_func = process_batch, 
    batch_type = OpenAIChatCompletionBatch, 
    batch_queue = batch_queue,
    batch_storage = batch_storage,
    wait_time = 3600 # check batches every 1 hour
)

request_kwargs = {
    "model": "gpt-4o-mini",
    "max_tokens": 1000,
    "temperature": 0.2
}
queue = InMemoryRequestQueue()
dispatcher = BatchDispatcher(
    batch_handler = handler, 
    queue = queue, 
    queue_threshold = 50000, # dispatch batches when the queue size >= queue_threshold
    time_threshold = 3600, # dispatch batch when the seconds since the last dispatch >= time_threshold
    # even if the queue size is less than the queue threshold
    time_interval = 600, # check the conditions every 600 seconds to dispatch batches
    requests_type = 'partial', # partial requests (only messages)
    request_kwargs = request_kwargs
)

# start the dispatcher and processor in the background
def run_dispatcher_in_background():
    loop = asyncio.get_event_loop()
    loop.create_task(dispatcher.run())

def run_processor_in_background():
    loop = asyncio.get_event_loop()
    loop.create_task(handler.run())

run_dispatcher_in_background()
run_processor_in_background()

# FastAPI API Service setup
app = FastAPI()

class MessagesList(BaseModel):
    data: List

async def handle_requests(messages_list: MessagesList):
    # add requests to the queue
    await asyncio.to_thread(dispatcher.queue.add_requests, list(messages_list.data))

@app.post("/requests")
async def handle_requests_api(messages_list: MessagesList):
    try:
        await handle_requests(messages_list)
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error handling requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

Now run command
```bash
uvicorn main:app --reload
```

Use `curl` to send a POST request to the API service
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/requests' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
        "data": [
            [
                {
                    "role": "user",
                    "content": "How can I learn Python?"
                }
            ],
            [
                {
                    "role": "user",
                    "content": "Who is the first president of the United States?"
                }
            ]
        ]
   }'
```