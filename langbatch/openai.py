from typing import Any, Dict
import tempfile
from pathlib import Path

import jsonlines
from openai import OpenAI, AzureOpenAI
from langbatch.Batch import Batch
from langbatch.schemas import OpenAIChatCompletionRequest, OpenAIEmbeddingRequest
from langbatch.ChatCompletionBatch import ChatCompletionBatch
from langbatch.EmbeddingBatch import EmbeddingBatch

class OpenAIBatch(Batch):
    """
    OpenAIBatch is a base class for OpenAI batch classes.
    Implements the Batch class for OpenAI API.
    """
    _url: str = "/v1/chat/completions"

    def __init__(self, file: str, client: OpenAI | AzureOpenAI = OpenAI()) -> None:
        """
        Initialize the OpenAIBatch class.

        Args:
            file (str): The path to the jsonl file in OpenAI batchformat.
            client (OpenAI, optional): The OpenAI client to use. Defaults to OpenAI().

        Usage:
        ```python
        batch = ChatCompletionBatch("path/to/file.jsonl")

        # With custom OpenAI client
        client = OpenAI(
            api_key="sk-proj-...",
            base_url="https://api.provider.com/v1"
        )
        batch = OpenAIBatch("path/to/file.jsonl", client = client)
        ```
        """
        super().__init__(file)
        self._client = client

    @classmethod
    def _get_init_args(cls, meta_data) -> Dict[str, Any]:
        return {}
    
    def _create_meta_data(self) -> Dict[str, Any]:
        return {}
    
    def _upload_batch_file(self):
        # Upload the batch file to OpenAI
        with open(self._file, "rb") as file:
            batch_input_file  = self._client.files.create(file=file, purpose="batch")
            return batch_input_file.id

    def _create_batch(self, input_file_id):
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint=self._url,
            completion_window= "24h"
        )
        self.platform_batch_id = batch.id

    def start(self):
        if self.platform_batch_id is not None:
            raise ValueError("Batch already started")
        
        batch_input_file_id = self._upload_batch_file()
        self._create_batch(batch_input_file_id)
    
    def cancel(self):
        """
        Usage:
        ```python
        # create a batch and start batch process
        batch = OpenAIChatCompletionBatch(file)
        batch.start()

        # cancel the batch process
        batch.cancel()
        ```
        """
        if self.platform_batch_id is None:
            raise ValueError("Batch not started")
        
        batch = self._client.batches.cancel(self.platform_batch_id)
        if batch.status == "cancelling" or batch.status == "cancelled":
            return True
        else:
            return False
        
    def get_status(self):
        if self.platform_batch_id is None:
            raise ValueError("Batch not started")
        
        batch = self._client.batches.retrieve(self.platform_batch_id)
        return batch.status

    def _download_results_file(self):
        batch_object = self._client.batches.retrieve(self.platform_batch_id)

        output_file_id = batch_object.output_file_id

        if output_file_id is not None:
            file_response = self._client.files.content(output_file_id).content
        else:
            return None  # Handle case where there's no output file
        
        file_path = self._create_results_file_path()
        with open(file_path, "wb") as file:
            file.write(file_response)

        error_file_id = batch_object.error_file_id
        if error_file_id is not None:
            error_file_response = self._client.files.content(error_file_id).content
            error_file_path = Path(tempfile.gettempdir()) / f"{self.id}_errors.jsonl"
            with open(error_file_path, "wb") as file:
                file.write(error_file_response)

            error_requests = []
            with jsonlines.open(error_file_path) as reader:
                for obj in reader:
                    error_requests.append(obj)

            with jsonlines.open(file_path, mode="a") as writer:
                writer.write_all(error_requests)

        return file_path
    
    def _get_errors(self):
        batch_object = self._client.batches.retrieve(self.platform_batch_id)
        return batch_object.errors
    
    def is_retryable_failure(self) -> bool:
        errors = self._get_errors()
        if errors:
            error = errors.data[0]['code']

            if error == "token_limit_exceeded":
                return True
            else:
                return False
        else:
            return False

    def retry(self):
        if self.platform_batch_id is None:
            raise ValueError("Batch not started")
        
        batch = self._client.batches.retrieve(self.platform_batch_id)

        self._create_batch(batch.input_file_id)

class OpenAIChatCompletionBatch(OpenAIBatch, ChatCompletionBatch):
    """
    OpenAIChatCompletionBatch is a class for OpenAI chat completion batches.
    Can be used for batch processing with gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4 models
    
    Usage:
    ```python
    batch = OpenAIChatCompletionBatch("path/to/file.jsonl")
    batch.start()
    ```
    """
    _url: str = "/v1/chat/completions"

    def _validate_request(self, request):
        OpenAIChatCompletionRequest(**request)

class OpenAIEmbeddingBatch(OpenAIBatch, EmbeddingBatch):
    """
    OpenAIEmbeddingBatch is a class for OpenAI embedding batches.
    Can be used for batch processing with text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 models

    Usage:
    ```python
    batch = OpenAIEmbeddingBatch("path/to/file.jsonl")
    batch.start()
    ```
    """
    _url: str = "/v1/embeddings"

    def _validate_request(self, request):
        OpenAIEmbeddingRequest(**request)