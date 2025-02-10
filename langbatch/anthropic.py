from typing import Any, Dict
import jsonlines
from anthropic import Anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

from langbatch.Batch import Batch
from langbatch.ChatCompletionBatch import ChatCompletionBatch
from langbatch.schemas import AnthropicChatCompletionRequest
from langbatch.claude_utils import convert_request, convert_response

anthropic_state_map = {
    'in_progress': 'in_progress',
    'succeeded': 'completed',
    'ended': 'completed',
    'errored': 'failed',
    'canceled': 'cancelled',
    'expired': 'expired',
}

class AnthropicBatch(Batch):
    """
    AnthropicBatch is a class for Anthropic batch processing.
    Implements the Batch class for Anthropic API.
    """
    _url: str = "https://api.anthropic.com/v1/messages/batches"

    def __init__(self, file: str, client: Anthropic = Anthropic()) -> None:
        """
        Initialize the AnthropicBatch class.

        Args:
            file (str): The path to the jsonl file in OpenAI batch format.
            client (Anthropic): The Anthropic client.

        Usage:
        ```python
        batch = AnthropicChatCompletionBatch(
            "path/to/file.jsonl"
        )
        ```
        """
        super().__init__(file)
        self._client = client
    
    def _create_meta_data(self) -> Dict[str, Any]:
        return {}

    def _upload_batch_file(self):
        pass

    def _get_init_args(self):
        return {}

    def _prepare_data(self):
        requests = self._get_requests()
        return [self._convert_request(request) for request in requests]
    
    def _create_batch(self):
        data = self._prepare_data()
        response = self._client.beta.messages.batches.create(
            requests=data,
        )
        self.platform_batch_id = response.id

    def start(self):
        if self.platform_batch_id is not None:
            raise ValueError("Batch already started")
        
        self._create_batch()
    
    def get_status(self):
        if self.platform_batch_id is None:
            raise ValueError("Batch not started")
        
        response = self._client.beta.messages.batches.retrieve(
            self.platform_batch_id
        )
        return anthropic_state_map[response.processing_status]

    def _download_results_file(self):
        if self.platform_batch_id is None:
            raise ValueError("Batch not started")
        
        file_path = self._create_results_file_path()
        with jsonlines.open(file_path, mode='w') as writer:
            for result in self._client.beta.messages.batches.results(
                self.platform_batch_id
            ):
                writer.write(self._convert_response(result.to_dict()))

        return file_path

    def _get_errors(self):
        # Implement error retrieval logic for Anthropic API
        batch = self._client.beta.messages.batches.retrieve(self.platform_batch_id)
        if batch.error:
            return batch.error.message
        else:
            return None
    
    def is_retryable_failure(self) -> bool:
        status = self.get_status()
        if status == "errored" or status == "expired":
            return True
        else:
            return False

    def retry(self):
        if self.platform_batch_id is None:
            raise ValueError("Batch not started")
        
        self._create_batch()

class AnthropicChatCompletionBatch(AnthropicBatch, ChatCompletionBatch):
    """
    AnthropicChatCompletionBatch is a class for Anthropic chat completion batches.
    
    Usage:
    ```python
    batch = AnthropicChatCompletionBatch("path/to/file.jsonl", client)
    batch.start()
    ```
    """
    def _convert_request(self, req: dict) -> Request:
        custom_id = req["custom_id"]
        request = convert_request(req)

        anthropic_request = Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(**request)
        )
        return anthropic_request
    
    def _convert_response(self, response) -> dict:
        return convert_response(response)

    def _validate_request(self, request):
        AnthropicChatCompletionRequest(**request)
