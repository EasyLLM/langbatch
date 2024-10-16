import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class BatchQueue(ABC):
    """
    Abstract class for batch queue.
    Implementations should provide a way to save and load batch queues.

    Used in BatchHandler to save and load the batch queues.

    Usage:
    ```python
    import asyncio

    # Using default FileBatchQueue
    file_batch_queue = FileBatchQueue("batch_queue.json")
    batch_handler = BatchHandler(
        batch_process_func=process_batch,
        batch_type=OpenAIChatCompletionBatch,
        batch_queue=file_batch_queue
    )
    asyncio.create_task(batch_handler.run())

    # With custom batch queue
    class MyCustomBatchQueue(BatchQueue):
        def save(self, queue: Dict[str, List[str]]):
            # Custom save logic

        def load(self) -> Dict[str, List[str]]:
            # Custom load logic

    custom_batch_queue = MyCustomBatchQueue()
    batch_handler = BatchHandler(
        batch_process_func=process_batch,
        batch_type=OpenAIChatCompletionBatch,
        batch_queue=custom_batch_queue
    )
    asyncio.create_task(batch_handler.run())
    ```
    """
    @abstractmethod
    def save(self, queue: Dict[str, List[str]]):
        """
        Save the batch queue.
        """
        pass

    @abstractmethod
    def load(self) -> Dict[str, List[str]]:
        """
        Load the batch queue.
        """
        pass

class FileBatchQueue(BatchQueue):
    """
    Batch queue that saves the queue to a file.

    Usage:
    ```python
    queue = FileBatchQueue("batch_queue.json")

    batch_handler = BatchHandler(
        batch_process_func=process_batch,
        batch_type=OpenAIChatCompletionBatch,
        batch_queue=queue
    )

    asyncio.create_task(batch_handler.run())
    ```
    """
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