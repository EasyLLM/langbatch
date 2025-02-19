import asyncio
import time
import logging
from typing import Dict, Literal
from langbatch.Batch import Batch
from langbatch.BatchHandler import BatchHandler
from langbatch.request_queues import RequestQueue
from langbatch.errors import BatchInitializationError

logger = logging.getLogger(__name__)

class BatchDispatcher:
    """
    Batch dispatcher creates batches from requests in the queue and dispatches them to the batch handler.
    It periodically checks the queue size and time threshold, and creates a batch and dispatches it to the batch handler.

    Usage:
    ```python
    # Create a batch dispatcher
    batch_dispatcher = BatchDispatcher(
        batch_handler=batch_handler,
        queue=request_queue,
        queue_threshold=50000,
        time_threshold=3600 * 2,
        requests_type="partial",
        request_kwargs=request_kwargs
    )

    asyncio.create_task(batch_dispatcher.run())
    ```
    """

    def __init__(
            self, 
            batch_handler: BatchHandler, 
            queue: RequestQueue, 
            queue_threshold: int = 50000, 
            time_threshold: int = 3600 * 2, 
            time_interval: int = 600, 
            requests_type: Literal["partial", "full"] = "partial", 
            request_kwargs: Dict = {}
        ):
        self.batch_handler = batch_handler
        self.queue = queue
        self.queue_threshold = queue_threshold
        self.time_threshold = time_threshold
        self.time_interval = time_interval
        self.last_batch_time = time.time()
        self.requests_type = requests_type
        self.request_kwargs = request_kwargs

    async def run(self):
        """
        Start the batch dispatcher as a asynchronous background task.
        Periodically checks the queue size and time threshold, and creates a batch and dispatches it to the batch handler.

        Examples:
            ```python
            asyncio.create_task(batch_dispatcher.run())
            ```
        """
        while True:
            logger.info("Running batch dispatcher")
            await self._check_batch_conditions()
            await asyncio.sleep(self.time_interval)

    async def _check_batch_conditions(self):
        logger.info("Checking queue for batch creation")
        while True:
            current_time = time.time()
            queue_size = len(self.queue)
            has_threshold_requests = queue_size >= self.queue_threshold
            reached_time_threshold = (current_time - self.last_batch_time) >= self.time_threshold
            if has_threshold_requests or (reached_time_threshold and queue_size > 0):
                logger.info("Creating and dispatching batch")
                await self._create_and_dispatch_batch()
            else:
                logger.info("No batch conditions met, waiting for next check")
                break

    async def _create_and_dispatch_batch(self):
        try:
            logger.info("Creating batch")
            requests = await asyncio.to_thread(self.queue.get_requests, self.queue_threshold)
            batch_class = self.batch_handler.batch_type
            batch_kwargs = self.batch_handler.batch_kwargs
            if self.requests_type == "partial":
                batch = await asyncio.to_thread(batch_class.create, requests, self.request_kwargs, batch_kwargs)
            else:
                batch = await asyncio.to_thread(batch_class.create_from_requests, requests, batch_kwargs)
            self.last_batch_time = time.time()
            await self._dispatch_batch(batch)
        except BatchInitializationError as e:
            logger.warning(f"Failed to create batch: {str(e)}")

    async def _dispatch_batch(self, batch: Batch):
        logger.info(f"Dispatching batch {batch.id}")
        await asyncio.to_thread(batch.save, self.batch_handler.batch_storage)
        
        await self.batch_handler.add_batch(batch.id)
        logger.info(f"Batch {batch.id} dispatched successfully")