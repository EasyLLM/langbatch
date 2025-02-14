import logging
from langbatch.BatchDispatcher import BatchDispatcher
from langbatch.BatchHandler import BatchHandler
from langbatch.factory import chat_completion_batch, embedding_batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%m-%y %H:%M:%S'
)

__all__ = ["chat_completion_batch", "embedding_batch"]