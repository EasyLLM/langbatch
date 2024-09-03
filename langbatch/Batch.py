"""
Batch class is the base class for all batch classes.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
from pathlib import Path

import jsonlines

class Batch(ABC):
    _url: str = ""

    def __init__(self, file) -> None:
        """
        Initialize the Batch class.
        """
        self._file = file # OpenAI compatible batch file in jsonl format
        self._id = uuid.uuid4()

        self._validate_requests() # Validate the requests in the batch file

    def _write_batch_file(self, requests) -> Path:
        """
        Write the batch file to the directory.
        """
        self._id = uuid.uuid4()

        # TODO: make the directory configurable
        file_path = Path(f"{self._id}.jsonl")
        with jsonlines.open(file_path, mode='w') as writer:
            writer.write_all(requests)

        return file_path

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
                except Exception as e:
                    print(f"Error processing item {item}: {e}")  # Logging the exception
                    continue

            file_path = cls._write_batch_file(requests)
        except:
            logging.error(f"Error writing batch file", exc_info=True)
            return None
        
        return file_path

    @abstractmethod
    def _upload_batch_file(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def cancel(self):
        pass

    @abstractmethod
    def status(self):
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
                self._validate_request(**request['body'])
            except:
                valid = False
           
            if not valid:
                invalid_requests.append(request['custom_id'])

        if len(invalid_requests) > 0:
            raise ValueError(f"Invalid requests: {invalid_requests}")
    
    # Retry on rate limit fail cases
    @abstractmethod
    def _retry(self):
        pass

    @abstractmethod
    def _download_results_file(self):
        pass
    
    # return results file in OpenAI compatible format
    @abstractmethod
    def get_results_file(self):
        pass

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