from openai import OpenAI
from langbatch.Batch import Batch
from langbatch.schemas import OpenAIChatCompletionRequest, OpenAIEmbeddingRequest
from langbatch.ChatCompletionBatch import ChatCompletionBatch
from langbatch.EmbeddingBatch import EmbeddingBatch

class OpenAIBatch(Batch):
    _url: str = "/v1/chat/completions"

    def __init__(self, file: str, client: OpenAI = OpenAI()) -> None:
        """
        Initialize the ChatCompletionBatch class.

        Args:
            file (str): The path to the jsonl file in OpenAI batchformat.
            client (OpenAI, optional): The OpenAI client to use. Defaults to OpenAI().

        Usage:
        ```python
        batch = ChatCompletionBatch("path/to/file.jsonl")

        # With custom OpenAI client
        client = OpenAI(
            api_key="sk-proj-...",
            base_url="https://api.llamas.exchange/v1"
        )
        batch = OpenAIBatch("path/to/file.jsonl", client = client)
        ```
        """
        super().__init__(file)
        self.client = client
        self.openai_batch_id = None

    def _upload_batch_file(self):
        # Upload the batch file to OpenAI
        with open(self._file, "rb") as file:
            batch_input_file  = self.client.files.create(file=file, purpose="batch")
            return batch_input_file.id
        
    def start(self):
        if self.openai_batch_id is not None:
            raise ValueError("Batch already started")
        
        batch_input_file_id = self._upload_batch_file()
        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=self._url,
            completion_window= "24h"
        )
        self.openai_batch_id = batch.id
    
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
        if self.openai_batch_id is None:
            raise ValueError("Batch not started")
        
        batch = self.client.batches.cancel(self.openai_batch_id)
        if batch.status == "cancelling" or batch.status == "cancelled":
            return True
        else:
            return False
        
    def status(self):
        if self.openai_batch_id is None:
            raise ValueError("Batch not started")
        
        batch = self.client.batches.retrieve(self.openai_batch_id)
        return batch.status

    def _download_results_file(self):
        batch_object = self.client.batches.retrieve(self.openai_batch_id)
        file_response = self.client.files.content(batch_object.output_file_id)
        
        file_path = f"{self.openai_batch_id}.jsonl"
        with open(file_path, "wb") as file:
            file.write(file_response.content)

        return file_path

    def get_results_file(self):
        file_path = self._download_results_file()

        return file_path
    
class OpenAIChatCompletionBatch(OpenAIBatch, ChatCompletionBatch):
    _url: str = "/v1/chat/completions"

    def _validate_request(self, request):
        OpenAIChatCompletionRequest(**request)

class OpenAIEmbeddingBatch(OpenAIBatch, EmbeddingBatch):
    _url: str = "/v1/embeddings"

    def _validate_request(self, request):
        OpenAIEmbeddingRequest(**request)