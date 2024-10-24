import logging
from typing import Dict, Callable, Type
from enum import Enum
import asyncio

from langbatch.Batch import Batch
from langbatch.batch_storages import BatchStorage, FileBatchStorage
from langbatch.batch_queues import BatchQueue, FileBatchQueue

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

class BatchHandler:
    """
    Batch handler that handles the batches in a queue manner. It handles:
    ```
    * starting batches
    * checking the status of batches
    * processing completed batches
    * retrying failed batches
    * cancelling non retryable failed batches
    ```

    Examples:
        ```python
        # Create a batch handler process
        batch_handler = BatchHandler(
            batch_process_func=process_batch,
            batch_type=OpenAIChatCompletionBatch
        )
        asyncio.create_task(batch_handler.run())

        # Add batches to the queue
        await batch_handler.add_batch("123")
        await batch_handler.add_batch("456")

        # With custom batch queue and batch storage
        custom_batch_queue = MyCustomBatchQueue()
        custom_batch_storage = MyCustomBatchStorage()
        batch_handler = BatchHandler(
            batch_process_func=process_batch,
            batch_type=OpenAIChatCompletionBatch,
            batch_queue=custom_batch_queue,
            batch_storage=custom_batch_storage
        )
        asyncio.create_task(batch_handler.run())
        ```
    """
    def __init__(
            self, 
            batch_process_func: Callable, 
            batch_type: Type[Batch], 
            batch_queue: BatchQueue = None,
            batch_storage: BatchStorage = None,
            wait_time: int = 3600,
            batch_kwargs: Dict = {}
        ):
        self.batch_process_func = batch_process_func
        self.batch_type = batch_type
        self.batch_queue = batch_queue or FileBatchQueue("batch_queue.json")
        self.queues = self.batch_queue.load()
        self.wait_time = wait_time
        self.batch_kwargs = batch_kwargs
        self.batch_storage = batch_storage or FileBatchStorage()

    async def add_batch(self, batch_id: str):
        """
        Add a batch to the queue.

        Parameters:
            batch_id: The ID of the batch to add.

        Examples:
            ```python
            await batch_handler.add_batch("123")
            ```
        """
        self.queues["pending"].append(batch_id)
        self._save_queues()
        logger.info(f"Added batch {batch_id} to pending queue")

    async def start_batch(self, batch: Batch):
        if batch.id in self.queues["pending"]:
            try:
                await asyncio.to_thread(batch.start)
                await asyncio.to_thread(batch.save, self.batch_storage)
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
                    logger.info(f"Processed batch {batch.id}")
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
                logger.info(f"Retrying batch {batch.id}")
                await asyncio.to_thread(batch.retry)
            except:
                logger.error(f"Error retrying batch {batch.id}", exc_info=True)
                await self.cancel_batch(batch.id)
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
        self.batch_queue.save(self.queues)

    async def run(self):
        """
        Start the batch handler as a asynchronous background task.
        Periodically checks the status of batches in the queue and processes them accordingly.

        Usage:
        ```python
        asyncio.create_task(batch_handler.run())
        ```
        """
        while True:
            logger.info("Handling batches")
            retried_batches = 0
            for batch_id in self.queues["processing"]:
                if self.batch_storage:
                    batch = self.batch_type.load(batch_id, storage = self.batch_storage)
                else:
                    batch = self.batch_type.load(batch_id)
                status = BatchStatus(await asyncio.to_thread(batch.get_status))
                
                if status == BatchStatus.COMPLETED:
                    await self.process_completed_batch(batch)
                elif status in [BatchStatus.FAILED, BatchStatus.EXPIRED]:
                    if retried_batches < 4:
                        retried = await self._handle_failed_or_expired_batch(batch, status)
                        if retried:
                            retried_batches += 1
                elif status in [BatchStatus.CANCELLING, BatchStatus.CANCELLED]:
                    await self.cancel_batch(batch_id)
                elif status not in [BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS, BatchStatus.FINALIZING]:
                    logger.error(f"Unknown status {status.value} for batch {batch_id}")
                    await self.cancel_batch(batch_id)

            if retried_batches < 4:
                started_batches = 0
                for batch_id in self.queues["pending"]:
                    if self.batch_storage:
                        batch = self.batch_type.load(batch_id, storage=self.batch_storage)
                    else:
                        batch = self.batch_type.load(batch_id)
                    await self.start_batch(batch)
                    started_batches += 1

                    if (started_batches + retried_batches) == 4:
                        break

            await asyncio.sleep(self.wait_time)

    async def _handle_failed_or_expired_batch(self, batch: 'Batch', status: BatchStatus):
        try:
            if status == BatchStatus.FAILED:
                retryable = await batch.is_retryable_failure()
                if retryable:
                    await asyncio.to_thread(self.retry_batch, batch)
                    return True
                else:
                    logger.warning(f"Batch {batch.id} failed due to non token-limit error")
                    await asyncio.to_thread(self.cancel_batch, batch.id)
                    return False
            elif status == BatchStatus.EXPIRED:
                await asyncio.to_thread(self.retry_batch, batch)
                return True
        except Exception as e:
            logger.error(f"Error handling {status.value} batch {batch.id}: {e}")
            return False