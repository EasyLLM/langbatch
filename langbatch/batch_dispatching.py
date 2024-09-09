import asyncio
import time
from typing import List, Any
import logging
from uuid import uuid4
from collections import deque

from langbatch.Batch import Batch
from langbatch.batch_processing import BatchHandler

logger = logging.getLogger(__name__)

class RequestQueue:
    def __init__(self):
        self.queue = deque()

    def add_requests(self, requests: List[Any]):
        self.queue.extend(requests)
        logger.info(f"Added {len(requests)} requests to queue. Queue size: {len(self.queue)}")

    def get_requests(self, count: int) -> List[Any]:
        if count > len(self.queue):
            count = len(self.queue)
        return [self.queue.popleft() for _ in range(count)]

    def __len__(self):
        return len(self.queue)

class BatchDispatcher:
    def __init__(self, batch_handler: BatchHandler, queue_threshold: int = 50000, time_threshold: int = 3600 * 2, **kwargs):
        self.batch_handler = batch_handler
        self.queue = RequestQueue()
        self.queue_threshold = queue_threshold
        self.time_threshold = time_threshold
        self.last_batch_time = time.time()
        self.kwargs = kwargs

    async def run(self):
        while True:
            logger.info("Running batch dispatcher")
            await self._check_batch_conditions()
            await asyncio.sleep(self.time_threshold)

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
            batch = await asyncio.to_thread(batch_class.create, requests, **self.kwargs)
            self.last_batch_time = time.time()
            await self._dispatch_batch(batch)
        except ValueError as e:
            logger.warning(f"Failed to create batch: {str(e)}")

    async def _dispatch_batch(self, batch: Batch):
        logger.info(f"Dispatching batch {batch.id}")
        await asyncio.to_thread(batch.save)
        await self.batch_handler.add_batch(batch.id)
        logger.info(f"Batch {batch.id} dispatched successfully")