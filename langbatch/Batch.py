"""
Batch class is the base class for all batch classes.
"""

import os
import json
import logging
import uuid
import shutil
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
from pathlib import Path

import jsonlines

# Default data path, can be overridden by environment variable
DEFAULT_DATA_PATH = Path(__file__).parent / "data"
DATA_PATH = DEFAULT_DATA_PATH

print("Hi World")

langbatch_data_path = os.environ.get("LANGBATCH_DATA_PATH")
if langbatch_data_path:
    try:
        data_path = Path(langbatch_data_path)
        test_file = data_path / "test.txt"
        # test if the directory is writable
        with open(test_file, 'w') as f:
            f.write("test")

        test_file.unlink(missing_ok=True)
        DATA_PATH = langbatch_data_path
    except:
        logging.warning(f"Invalid data path: {langbatch_data_path}, using default data path: {DEFAULT_DATA_PATH}")
    
class BatchStorage(ABC):
    @abstractmethod
    def save(self, id: str, data_file: Path, meta_data: Dict[str, Any]):
        pass

    @abstractmethod
    def load(self, id: str) -> Tuple[Path, Path]:
        pass

class FileBatchStorage(BatchStorage):
    def __init__(self, directory: str = DATA_PATH):
        self.saved_batches_directory = Path(directory) / "saved_batches"
        self.saved_batches_directory.mkdir(exist_ok=True, parents=True)

    def save(self, id: str, data_file: Path, meta_data: Dict[str, Any]):
        """
        Save the batch data and metadata to the storage.
        """
        with open(self.saved_batches_directory / f"{id}.json", 'w') as f:
            json.dump(meta_data, f)

        destination = self.saved_batches_directory / f"{id}.jsonl"
        if not destination.exists(): 
            # if the file does not exist, copy the file from the data_file
            shutil.copy(data_file, destination)

    def load(self, id: str) -> Tuple[Path, Path]:
        data_file = self.saved_batches_directory / f"{id}.jsonl"
        json_file = self.saved_batches_directory / f"{id}.json"

        if not data_file.exists() or not json_file.exists():
            raise ValueError(f"Batch with id {id} not found")
        
        return data_file, json_file

class Batch(ABC):
    _url: str = ""
    platform_batch_id: str | None = None

    def __init__(self, file: str) -> None:
        """
        Initialize the Batch class.
        """
        self._file = file # OpenAI compatible batch file in jsonl format
        self.id = str(uuid.uuid4())

        self._validate_requests() # Validate the requests in the batch file

    @classmethod
    def _create_batch_file(cls, key: str, data: List[Any], **kwargs) -> Path | None:
        """
        Create the batch file when given a list of items.
        For Chat Completions, this would be a list of messages.
        For Embeddings, this would be a list of texts.

        kwargs is used to pass in the parameters for the API call. 
        Ex. model, temperature, etc.
        """
        requests = []
        try:
            for item in data:
                try:
                    body = kwargs.copy()  # Copy kwargs to avoid mutation
                    custom_id = str(uuid.uuid4())

                    body[key] = item
                    
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": cls._url,
                        "body": body
                    }
                    requests.append(request)
                except:
                    logging.warning(f"Error processing item {item}", exc_info= True)
                    continue

            batches_dir = Path(DATA_PATH) / "created_batches"
            batches_dir.mkdir(exist_ok=True, parents=True)

            id = str(uuid.uuid4())
            file_path = batches_dir / f"{id}.jsonl"
            with jsonlines.open(file_path, mode='w') as writer:
                writer.write_all(requests)
        except:
            logging.error(f"Error writing batch file", exc_info=True)
            return None
        
        if file_path is None:
            raise ValueError("Failed to create batch. Check the input data.")
        
        return cls(file_path)

    @abstractmethod
    def _upload_batch_file(self):
        pass

    @classmethod
    def load(cls, id: str, storage: BatchStorage = FileBatchStorage()):
        """
        Load a batch from the storage and return a Batch object.

        Args:
            id (str): The id of the batch.
            storage (BatchStorage, optional): The storage to load the batch from. Defaults to FileBatchStorage().

        Returns:
            Batch: The batch object.

        Usage:
        ```python
        batch = OpenAIChatCompletionBatch.load("123", storage=FileBatchStorage("./data"))
        ```
        """
        data_file, json_file = storage.load(id)

        batch = cls(str(data_file))

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        batch.platform_batch_id = json_data['platform_batch_id']
        batch.id = id

        return batch

    def save(self, storage: BatchStorage = FileBatchStorage()):
        """
        Save the batch to the storage.

        Args:
            storage (BatchStorage, optional): The storage to save the batch to. Defaults to FileBatchStorage().

        Usage:
        ```python
        batch = OpenAIChatCompletionBatch(file)
        batch.save()

        # save the batch to file storage
        batch.save(storage=FileBatchStorage("./data"))
        ```
        """
        meta_data = {
            "platform_batch_id": self.platform_batch_id
        }
        storage.save(self.id, Path(self._file), meta_data)

    @abstractmethod
    def start(self):
        """
        Usage:
        ```python
        # create a batch
        batch = OpenAIChatCompletionBatch(file)

        # start the batch process
        batch.start()
        ```
        """
        pass

    @abstractmethod
    def status(self):
        """
        Usage:
        ```python
        # create a batch and start batch process
        batch = OpenAIChatCompletionBatch(file)
        batch.start()

        # get the status of the batch process
        status = batch.status()
        print(status)
        ```
        """
        pass

    def _get_requests(self) -> List[Dict[str, Any]]:
        """
        Get all the requests from the jsonl batch file.
        """
        requests = []
        try:
            with jsonlines.open(self._file) as reader:
                for obj in reader:
                    requests.append(obj)
        except:
            logging.error(f"Error reading requests from batch file", exc_info=True)
            raise ValueError("Error reading requests from batch file")

        return requests

    @abstractmethod
    def _validate_request(self, request):
        pass

    def _validate_requests(self) -> None:
        """
        Validate all the requests in the batch file before starting the batch process.

        Depends on the implementation of the _validate_request method in the subclass.
        """
        invalid_requests = []
        for request in self._get_requests():
            valid = True
            try:
                self._validate_request(request['body'])
            except:
                logging.info(f"Invalid request: {request}", exc_info=True)
                valid = False
           
            if not valid:
                invalid_requests.append(request['custom_id'])

        if len(invalid_requests) > 0:
            raise ValueError(f"Invalid requests: {invalid_requests}")
    
    @abstractmethod
    def _download_results_file(self):
        pass
    
    # return results file in OpenAI compatible format
    @abstractmethod
    def get_results_file(self):
        """
        Usage:
        ```python
        import jsonlines

        # create a batch and start batch process
        batch = OpenAIChatCompletionBatch(file)
        batch.start()

        if batch.status() == "completed":
            # get the results file
            results_file = batch.get_results_file()

            with jsonlines.open(results_file) as reader:
                for obj in reader:
                    print(obj)
        ```
        """

    def _prepare_results(
        self, process_func
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]] | Tuple[None, None]:
        """
        Prepare the results file by processing the results,
        and separating them into successful and unsuccessful results
        based on the status code of the response.

        Depends on the implementation of the process_func method in the subclass.
        """

        file_id = self._download_results_file()

        if file_id is None:
            return None, None

        try:
            results = []
            with jsonlines.open(file_id) as reader:
                for obj in reader:
                    results.append(obj)

            successful_results = []
            unsuccessful_results = []
            for result in results:
                if result['response']['status_code'] == 200:
                    choices = {
                        "custom_id": result['custom_id'],
                        **process_func(result)
                    }
                    successful_results.append(choices)
                else:
                    error = {
                        "custom_id": result['custom_id'],
                        "error": result['error']
                    }
                    unsuccessful_results.append(error)

            return successful_results, unsuccessful_results
        except:
            logging.error(f"Error preparing results file", exc_info=True)
            return None, None
    
    # return results list
    @abstractmethod
    def get_results(self):
        pass

    @abstractmethod
    def is_retryable_failure(self) -> bool:
        pass

    # Retry on rate limit fail cases
    @abstractmethod
    def retry(self):
        pass