"""
Batch class is the base class for all batch classes.
"""

import json
import logging
import uuid
import pickle
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
from pathlib import Path

import jsonlines
from langbatch.batch_storages import DATA_PATH, BatchStorage, FileBatchStorage
from langbatch.errors import BatchInitializationError, BatchError, BatchValidationError

class Batch(ABC):
    """
    Batch class is the base class for all batch classes.

    Implementations of this class will be platform specific (OpenAI, Vertex AI, etc.)
    """
    _url: str = ""
    platform_batch_id: str | None = None

    def __init__(self, file: str):
        """
        Initialize the Batch class.

        Args:
            file (str): The path to the batch file. File should be in OpenAI compatible batch file in jsonl format.
        """
        self._file = file
        self.id = str(uuid.uuid4())

        self._validate_requests() # Validate the requests in the batch file

    @classmethod
    def _create_batch_file_from_requests(cls, requests) -> Path:
        try:
            batches_dir = Path(DATA_PATH) / "created_batches"
            batches_dir.mkdir(exist_ok=True, parents=True)

            id = str(uuid.uuid4())
            file_path = batches_dir / f"{id}.jsonl"
            with jsonlines.open(file_path, mode='w') as writer:
                writer.write_all(requests)
        except:
            logging.error(f"Error creating batch file", exc_info=True)
            return None

        return file_path

    @classmethod
    def _create_batch_file(cls, key: str, data: List[Any], request_kwargs: Dict = {}, batch_kwargs: Dict = {}) -> Path | None:
        """
        Create the batch file when given a list of items.
        For Chat Completions, this would be a list of messages.
        For Embeddings, this would be a list of texts.
        """
        requests = []
        try:
            for item in data:
                try:
                    body = request_kwargs.copy()  # Copy kwargs to avoid mutation
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
        except:
            logging.error(f"Error creating requests from data to create batch file", exc_info=True)
            return None
        
        file_path = cls._create_batch_file_from_requests(requests)
        
        if file_path is None:
            raise BatchInitializationError("Failed to create batch. Check the input data.")
        
        return cls(file_path, **batch_kwargs)

    @classmethod
    def create_from_requests(cls, requests, batch_kwargs: Dict = {}):
        """
        Creates a batch when given a list of requests. 
        These requests should be in correct Batch API request format as per the Batch type.
        Ex. for OpenAIChatCompletionBatch, requests should be a Chat Completion request with custom_id.

        Args:
            requests: A list of requests.
            batch_kwargs (Dict, optional): Additional keyword arguments for the batch class. Ex. gcp_project, etc. for VertexAIChatCompletionBatch.

        Returns:
            An instance of the Batch class.

        Raises:
            BatchInitializationError: If the input data is invalid.

        Usage:
        ```python
        batch = OpenAIChatCompletionBatch.create_from_requests([
            {   "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Biryani Receipe, pls."}],
                    "max_tokens": 1000
                }
            },
            {
                "custom_id": "request-2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Write a short story about AI"}],
                    "max_tokens": 1000
                }
            }
        ]
        ``` 
        """

        file_path = cls._create_batch_file_from_requests(requests)

        if file_path is None:
            raise BatchInitializationError("Failed to create batch. Check the input data.")
        
        return cls(file_path, **batch_kwargs)

    @classmethod
    @abstractmethod
    def _get_init_args(cls, meta_data) -> Dict[str, Any]:
        """
        Get the init arguments from meta data json file when loading a batch from storage.
        """
        pass

    @classmethod
    def load(cls, id: str, storage: BatchStorage = FileBatchStorage(), batch_kwargs: Dict = {}):
        """
        Load a batch from the storage and return a Batch object.

        Args:
            id (str): The id of the batch.
            storage (BatchStorage, optional): The storage to load the batch from. Defaults to FileBatchStorage().
            batch_kwargs (Dict, optional): Additional keyword arguments for the batch class. Ex. gcp_project, etc. for VertexAIChatCompletionBatch.

        Returns:
            Batch: The batch object.

        Usage:
        ```python
        batch = OpenAIChatCompletionBatch.load("123", storage=FileBatchStorage("./data"))
        ```
        """
        data_file, meta_file = storage.load(id)

        # Load metadata based on file extension
        if meta_file.suffix == '.json':
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
        else:  # .pkl
            with open(meta_file, 'rb') as f:
                meta_data = pickle.load(f)

        init_args = cls._get_init_args(meta_data)

        for key, value in batch_kwargs.items():
            if key not in init_args:
                init_args[key] = value

        batch = cls(str(data_file), **init_args)
        batch.platform_batch_id = meta_data['platform_batch_id']
        batch.id = id

        return batch
    
    @abstractmethod
    def _create_meta_data(self) -> Dict[str, Any]:
        """
        Create the meta data for the batch to be saved in the storage.
        """
        pass

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
        meta_data = self._create_meta_data()
        meta_data["platform_batch_id"] = self.platform_batch_id

        storage.save(self.id, Path(self._file), meta_data)

    @abstractmethod
    def _upload_batch_file(self):
        pass

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
    def get_status(self):
        """
        Usage:
        ```python
        # create a batch and start batch process
        batch = OpenAIChatCompletionBatch(file)
        batch.start()

        # get the status of the batch process
        status = batch.get_status()
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
            raise BatchError("Error reading requests from batch file")

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
            raise BatchValidationError(f"Invalid requests: {invalid_requests}")
        
        if len(self._get_requests()) == 0:
            raise BatchValidationError("No requests found in the batch file")
    
    def _create_results_file_path(self):
        results_dir = Path(DATA_PATH) / "results"
        results_dir.mkdir(exist_ok=True)

        return results_dir / f"{self.id}.jsonl"

    @abstractmethod
    def _download_results_file(self):
        pass
    
    # return results file in OpenAI compatible format
    def get_results_file(self):
        """
        Usage:
        ```python
        import jsonlines

        # create a batch and start batch process
        batch = OpenAIChatCompletionBatch(file)
        batch.start()

        if batch.get_status() == "completed":
            # get the results file
            results_file = batch.get_results_file()

            with jsonlines.open(results_file) as reader:
                for obj in reader:
                    print(obj)
        ```
        """
        file_path = self._download_results_file()
        return file_path

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
                if result['response'] is None:
                    if result['error'] is not None:
                        error = {
                            "custom_id": result['custom_id'],
                            "error": result['error']
                        }
                    else:
                        error = {
                            "custom_id": result['custom_id'],
                            "error": "No response from the API"
                        }
                    unsuccessful_results.append(error)
                    continue

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

    def get_unsuccessful_requests(self) -> List[Dict[str, Any]]:
        """
        Retrieve the unsuccessful requests from the batch.

        Returns:
            A list of requests that failed.

        Usage:
        ```python
        batch = OpenAIChatCompletionBatch(file)
        batch.start()

        if batch.get_status() == "completed":
            # get the unsuccessful requests
            unsuccessful_requests = batch.get_unsuccessful_requests()

            for request in unsuccessful_requests:
                print(request["custom_id"])
        ```
        """
        custom_ids = []
        _, unsuccessful_results = self.get_results()
        for result in unsuccessful_results:
            custom_ids.append(result["custom_id"])
        
        return self.get_requests_by_custom_ids(custom_ids)

    def get_requests_by_custom_ids(self, custom_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve the requests from the batch file by custom ids.

        Args:
            custom_ids (List[str]): A list of custom ids.

        Returns:
            A list of requests.

        Usage:
        ```python
        batch = OpenAIChatCompletionBatch(file)
        batch.start()

        if batch.get_status() == "completed":
            # get the requests by custom ids
            requests = batch.get_requests_by_custom_ids(["custom_id1", "custom_id2"])

            for request in requests:
                print(request["custom_id"])
        ```
        """
        requests = []
        with jsonlines.open(self._file) as reader:
            for request in reader:
                if request["custom_id"] in custom_ids:
                    requests.append(request)
        return requests