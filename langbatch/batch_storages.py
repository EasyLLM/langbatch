import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple
import pickle
from langbatch.utils import get_data_path

DATA_PATH = get_data_path()

def _is_json_serializable(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError, ValueError):
        return False

class BatchStorage(ABC):
    """
    Abstract class for batch storage.
    Implementations should provide a way to save and load batches.

    Usage:
    ```python
    batch = OpenAIChatCompletionBatch("path/to/file.jsonl")
    batch.start()

    # Using default FileBatchStorage
    batch.save(storage=FileBatchStorage()) # same as batch.save()
    batch = OpenAIChatCompletionBatch.load("1ff73c3f", storage=FileBatchStorage()) # same as OpenAIChatCompletionBatch.load("1ff73c3f")

    # With custom storage
    class MyCustomBatchStorage(BatchStorage):
        def save(self, id: str, data_file: Path, meta_data: Dict[str, Any]):
            # Custom save logic

        def load(self, id: str) -> Tuple[Path, Path]:
            # Custom load logic
        
    custom_storage = MyCustomBatchStorage()

    # Using custom storage
    batch.save(storage=custom_storage)
    batch = OpenAIChatCompletionBatch.load("1ff73c3f", storage=custom_storage)
    ```
    """

    @abstractmethod
    def save(self, id: str, data_file: Path, meta_data: Dict[str, Any]):
        """
        Save the batch data and metadata to the storage.

        Args:
            id (str): The id of the batch.
            data_file (Path): The path to the batch data file.
            meta_data (Dict[str, Any]): The metadata of the batch.
        """
        pass

    @abstractmethod
    def load(self, id: str) -> Tuple[Path, Path]:
        """
        Load the batch data and metadata from the storage.

        Args:
            id (str): The id of the batch.

        Returns:
            Tuple[Path, Path]: The path to the batch data jsonlfile and the path to the metadata jsonfile.
        """
        pass

class FileBatchStorage(BatchStorage):
    """
    Batch storage that saves the batch data and metadata to the file system.
    Automatically chooses between JSON and pickle serialization based on content:
    - Uses JSON for simple metadata (human-readable, portable)
    - Uses pickle for complex objects (like API clients)

    Usage:
    ```python
    batch = OpenAIChatCompletionBatch("path/to/file.jsonl")
    batch.start()

    # Save the batch
    batch.save(storage=FileBatchStorage())

    # Load the batch
    batch = OpenAIChatCompletionBatch.load("1ff73c3f", storage=FileBatchStorage())
    ```
    """

    def __init__(self, directory: str = DATA_PATH):
        """
        Initialize the FileBatchStorage. Will create or use a directory named 'saved_batches' in the given directory to save the batches.

        Args:
            directory (str): The directory to save the batches. Defaults to the DATA_PATH.
        """
        self.saved_batches_directory = Path(directory) / "saved_batches"
        self.saved_batches_directory.mkdir(exist_ok=True, parents=True)

    def save(self, id: str, data_file: Path, meta_data: Dict[str, Any]):
        # Check if metadata can be JSON serialized
        use_json = _is_json_serializable(meta_data)
        json_meta_file = self.saved_batches_directory / f"{id}.json"
        pkl_meta_file = self.saved_batches_directory / f"{id}.pkl"
        if use_json:
            with open(json_meta_file, 'w') as f:
                json.dump(meta_data, f)

            if pkl_meta_file.exists():
                pkl_meta_file.unlink(missing_ok=True)
        else:
            with open(pkl_meta_file, 'wb') as f:
                pickle.dump(meta_data, f)

            if json_meta_file.exists():
                json_meta_file.unlink(missing_ok=True)

        destination = self.saved_batches_directory / f"{id}.jsonl"
        if not destination.exists(): 
            # if the file does not exist, copy the file from the data_file
            shutil.copy(data_file, destination)

    def load(self, id: str) -> Tuple[Path, Path]:
        data_file = self.saved_batches_directory / f"{id}.jsonl"
        
        # Try JSON first, then pickle
        json_file = self.saved_batches_directory / f"{id}.json"
        pkl_file = self.saved_batches_directory / f"{id}.pkl"
        
        if json_file.is_file():
            meta_file = json_file
        elif pkl_file.is_file():
            meta_file = pkl_file
        else:
            raise ValueError(f"Batch with id {id} not found")

        if not data_file.is_file():
            raise ValueError(f"Batch with id {id} not found")
        
        return data_file, meta_file