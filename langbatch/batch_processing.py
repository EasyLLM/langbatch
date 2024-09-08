import logging
import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum
import time

from langbatch.openai import Batch

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

class BatchQueueManager:
    def __init__(self, batch_process_func, storage: BatchQueueStorage):
        self.batch_process_func = batch_process_func
        self.storage = storage
        self.queues = self.storage.load()

    def add_batch(self, batch_id: str):
        self.queues["pending"].append(batch_id)
        self._save_queues()
        logger.info(f"Added batch {batch_id} to pending queue")

    def start_batch(self, batch: Batch):
        if batch.id in self.queues["pending"]:
            try:
                batch.start()
                self.queues["processing"].append(batch.id)
                logger.info(f"Moved batch {batch.id} from pending to processing queue")
            except:
                logger.error(f"Error starting batch {batch.id}", exc_info=True)
            finally:
                self.queues["pending"].remove(batch.id)
            
            self._save_queues() 
        else:
            logger.warning(f"Batch {batch.id} not found in pending queue")

    def process_completed_batch(self, batch: Batch):
        if batch.id in self.queues["processing"]:
            try:
                self.batch_process_func(batch)
            except:
                logger.error(f"Error processing completed batch {batch.id}", exc_info=True)
            self.queues["processing"].remove(batch.id)
            self._save_queues()
            logger.info(f"Removed completed batch {batch.id} from processing queue")
        else:
            logger.warning(f"Completed batch {batch.id} not found in processing queue")

    def retry_batch(self, batch: Batch):
        if batch.id in self.queues["processing"]:
            try:
                batch.retry()
            except:
                logger.error(f"Error retrying batch {batch.id}", exc_info=True)
                self.cancel_batch(batch.id)
        else:
            logger.warning(f"Batch {batch.id} not found in processing queue for retry")

    def cancel_batch(self, batch_id: str):
        for queue in self.queues.values():
            if batch_id in queue:
                queue.remove(batch_id)
                self._save_queues()
                logger.info(f"Cancelled and removed batch {batch_id} from queue")
                return
        logger.warning(f"Batch {batch_id} not found in any queue for cancellation")

    def _save_queues(self):
        self.storage.save(self.queues)

class BatchProcessor:
    def __init__(self, queue_manager: BatchQueueManager):
        self.queue_manager = queue_manager

    def process_batches(self):
        while True:
            retried_batches = 0
            for batch_id in self.queue_manager.queues["processing"]:
                batch = Batch.load(batch_id)
                status = batch.status()

                if status == BatchStatus.COMPLETED:
                    self._process_completed_batch(batch)
                elif status == BatchStatus.FAILED or status == BatchStatus.EXPIRED:
                    if retried_batches < 4:
                        if status == BatchStatus.FAILED:    
                            retried = self._handle_failed_batch(batch)
                        else:
                            retried = self._handle_expired_batch(batch)

                        if retried:
                            retried_batches += 1
                elif status in [BatchStatus.CANCELLING, BatchStatus.CANCELLED]:
                    self.queue_manager.cancel_batch(batch_id)

            if retried_batches < 4:
                started_batches = 0
                for batch_id in self.queue_manager.queues["pending"]:
                    batch = Batch.load(batch_id)
                    self.queue_manager.start_batch(batch)
                    started_batches += 1

                    if (started_batches + retried_batches) == 4:
                        break

            time.sleep(1800)  # Check every minute

    def _process_completed_batch(self, batch: Batch):
        try:
            logger.info(f"Processing completed batch {batch.id}")
            self.queue_manager.process_completed_batch(batch.id)
        except Exception as e:
            logger.error(f"Error processing completed batch {batch.id}: {e}")

    def _handle_failed_batch(self, batch: Batch):
        try:
            if batch.is_retryable_failure():
                self.queue_manager.retry_batch(batch.id)
                return True
            else:
                logger.warning(f"Batch {batch.id} failed due to non-token-limit error")
                self.queue_manager.cancel_batch(batch.id)
                return False
        except Exception as e:
            logger.error(f"Error handling failed batch {batch.id}: {e}")
            return False

    def _handle_expired_batch(self, batch: Batch):
        try:
            logger.info(f"Processing expired batch {batch.id}")
            self.queue_manager.retry_batch(batch.id)
            return True
        except Exception as e:
            logger.error(f"Error processing expired batch {batch.id}: {e}")
            return False

