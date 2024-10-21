import logging
from langbatch.BatchDispatcher import BatchDispatcher
from langbatch.BatchHandler import BatchHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%m-%y %H:%M:%S'
)