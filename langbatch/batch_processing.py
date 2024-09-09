import logging
import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Type
from enum import Enum
import asyncio

from langbatch.Batch import Batch

logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

class BatchQueueStorage(ABC):
    @abstractmethod
    def save(self, queue: Dict[str, List[str]]):
        pass

    @abstractmethod
    def load(self) -> Dict[str, List[str]]:
        pass

class FileBatchQueueStorage(BatchQueueStorage):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save(self, queue: Dict[str, List[str]]):
        try:
            with open(self.file_path, 'w') as f:
                json.dump(queue, f)
        except IOError as e:
            logger.error(f"Error saving queue to file: {e}")
            raise

    def load(self) -> Dict[str, List[str]]:
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            return {"pending": [], "processing": []}
        except IOError as e:
            logger.error(f"Error loading queue from file: {e}")
            raise

class BatchHandler:
    def __init__(
            self, 
            batch_process_func: Callable, 
            batch_type: Type[Batch], 
            storage: BatchQueueStorage = None,
            wait_time: int = 3600
        ):
        self.batch_process_func = batch_process_func
        self.batch_type = batch_type
        self.storage = storage or FileBatchQueueStorage("batch_queue.json")
        self.queues = self.storage.load()
        self.wait_time = wait_time

    async def add_batch(self, batch_id: str):
        self.queues["pending"].append(batch_id)
        self._save_queues()
        logger.info(f"Added batch {batch_id} to pending queue")

    async def start_batch(self, batch: Batch):
        if batch.id in self.queues["pending"]:
            try:
                await asyncio.to_thread(batch.start)
                self.queues["processing"].append(batch.id)
                logger.info(f"Moved batch {batch.id} from pending to processing queue")
            except:
                logger.error(f"Error starting batch {batch.id}", exc_info=True)
            finally:
                self.queues["pending"].remove(batch.id)
            
            self._save_queues() 
        else:
            logger.warning(f"Batch {batch.id} not found in pending queue")

    async def process_completed_batch(self, batch: Batch):
        try:
            logger.info(f"Processing completed batch {batch.id}")
            if batch.id in self.queues["processing"]:
                try:
                    await asyncio.to_thread(self.batch_process_func, batch)
                except:
                    logger.error(f"Error processing completed batch {batch.id}", exc_info=True)
                self.queues["processing"].remove(batch.id)
                self._save_queues()
                logger.info(f"Removed completed batch {batch.id} from processing queue")
            else:
                logger.warning(f"Completed batch {batch.id} not found in processing queue")
        except:
            logger.error(f"Error processing completed batch {batch.id}", exc_info=True)

    async def retry_batch(self, batch: Batch):
        if batch.id in self.queues["processing"]:
            try:
                await asyncio.to_thread(batch.retry)
            except:
                logger.error(f"Error retrying batch {batch.id}", exc_info=True)
                self.cancel_batch(batch.id)
        else:
            logger.warning(f"Batch {batch.id} not found in processing queue for retry")

    async def cancel_batch(self, batch_id: str):
        for queue in self.queues.values():
            if batch_id in queue:
                queue.remove(batch_id)
                self._save_queues()
                logger.info(f"Cancelled and removed batch {batch_id} from queue")
                return
        logger.warning(f"Batch {batch_id} not found in any queue for cancellation")

    def _save_queues(self):
        self.storage.save(self.queues)

    async def run(self):
        while True:
            logger.info("Handling batches")
            retried_batches = 0
            for batch_id in self.queues["processing"]:
                batch = self.batch_type.load(batch_id)
                status = BatchStatus(await asyncio.to_thread(batch.status))
                
                if status == BatchStatus.COMPLETED:
                    await self.process_completed_batch(batch)
                elif status in [BatchStatus.FAILED, BatchStatus.EXPIRED]:
                    if retried_batches < 4:
                        retried = await self._handle_failed_or_expired_batch(batch, status)
                        if retried:
                            retried_batches += 1
                elif status in [BatchStatus.CANCELLING, BatchStatus.CANCELLED]:
                    await self.cancel_batch(batch_id)

            if retried_batches < 4:
                started_batches = 0
                for batch_id in self.queues["pending"]:
                    batch = self.batch_type.load(batch_id)
                    await self.start_batch(batch)
                    started_batches += 1

                    if (started_batches + retried_batches) == 4:
                        break

            await asyncio.sleep(self.wait_time)

    async def _handle_failed_or_expired_batch(self, batch: 'Batch', status: BatchStatus):
        try:
            if status == BatchStatus.FAILED:
                if batch.is_retryable_failure():
                    await asyncio.to_thread(self.retry_batch, batch)
                    return True
                else:
                    logger.warning(f"Batch {batch.id} failed due to non-token-limit error")
                    await asyncio.to_thread(self.cancel_batch, batch.id)
                    return False
            elif status == BatchStatus.EXPIRED:
                logger.info(f"Processing expired batch {batch.id}")
                await asyncio.to_thread(self.retry_batch, batch)
                return True
        except Exception as e:
            logger.error(f"Error handling {status.value} batch {batch.id}: {e}")
            return False

