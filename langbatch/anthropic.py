import time
from typing import Any, Dict, List, Optional
import jsonlines
from anthropic import Anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

from langbatch.Batch import Batch
from langbatch.ChatCompletionBatch import ChatCompletionBatch
from langbatch.schemas import AnthropicChatCompletionRequest
from langbatch.utils import get_web_image

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
        self.client = client
    
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
        response = self.client.beta.messages.batches.create(
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
        
        response = self.client.beta.messages.batches.retrieve(
            self.platform_batch_id
        )
        return anthropic_state_map[response.processing_status]

    def _download_results_file(self):
        if self.platform_batch_id is None:
            raise ValueError("Batch not started")
        
        file_path = self._create_results_file_path()
        with jsonlines.open(file_path, mode='w') as writer:
            for result in self.client.beta.messages.batches.results(
                self.platform_batch_id
            ):
                writer.write(self._convert_response(result))

        return file_path

    def _get_errors(self):
        # Implement error retrieval logic for Anthropic API
        batch = self.client.beta.messages.batches.retrieve(self.platform_batch_id)
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
    batch = AnthropicChatCompletionBatch("path/to/file.jsonl", "claude-3-sonnet-20240229", "your-api-key")
    batch.start()
    ```
    """
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted_messages = []
        for message in messages:
            if message["role"] == "assistant" and message["tool_calls"]:
                converted_tool_calls = []
                for tool_call in message["tool_calls"]:
                    converted_tool_call = {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": tool_call["function"]["arguments"]
                    }
                    converted_tool_calls.append(converted_tool_call)
                converted_message = {"role": "assistant", "content": [converted_tool_call]}
            elif message["role"] == "tool":
                converted_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message["content"]
                        }
                    ]
                }
            else:
                converted_message = {
                    "role": message["role"],
                    "content": self._convert_content(message["content"])
                }
            converted_messages.append(converted_message)
        return converted_messages

    def _convert_content(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        elif isinstance(content, list):
            converted_content = []
            for item in content:
                if isinstance(item, str):
                    converted_content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if item["type"] == "text":
                        converted_content.append(item)
                    elif item["type"] == "image_url":
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:"):
                            image_media_type = image_url.split(";")[0].split(":")[-1]
                            image_data = image_url.split(",")[1]
                        else:
                            image_media_type, image_data = get_web_image(image_url)

                        converted_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_data
                            }
                        })

            return converted_content
        return []

    def _convert_tools(self, tools: Optional[List[Dict[str, Any]]]):
        if not tools:
            return None
        
        converted_tools = []
        for tool in tools:
            if tool["type"] == "function":
                converted_tool = {
                    "name": tool["function"]["name"],
                    "input_schema": tool["function"]["parameters"]
                }
                if tool["function"]["description"]:
                    converted_tool["description"] = tool["function"]["description"]
                converted_tools.append(converted_tool)
        return converted_tools

    def _convert_tool_choice(self, tools_given: bool, tool_choice: Optional[Dict[str, Any]], parallel_tool_calls: Optional[bool]):
        tool_choice_obj = None
        if tool_choice is None and tools_given:
            tool_choice_obj = {"type": "auto"}
        
        if isinstance(tool_choice, str):
            match tool_choice:
                case "auto":
                    tool_choice_obj = {"type": "auto"}
                case "required":
                    tool_choice_obj = {"type": "any"}
                case "none":
                    tool_choice_obj = {"type": "auto"} if tools_given else None
        elif isinstance(tool_choice, dict):
            if tool_choice["type"] == "function":
                return {"type": "tool", "name": tool_choice["function"]["name"]}
        
        # Handle parallel_tool_calls
        if parallel_tool_calls and tool_choice_obj:
            tool_choice_obj["disable_parallel_tool_use"] = parallel_tool_calls
        
        return tool_choice_obj

    def _convert_request(self, req: dict) -> Request:
        custom_id = req["custom_id"]
        request = AnthropicChatCompletionRequest(**req["body"])

        messages = []
        system = ""
        for message in request.messages:
            if message["role"] == "system":
                if isinstance(message["content"], str):
                    system = message["content"]
                elif isinstance(message["content"], dict):
                    try:
                        system = message["content"]["text"]
                    except KeyError:
                        pass
            else:
                messages.append(message)

        messages = self._convert_messages(messages)

        req = {
            "model": request.model,
            "messages": messages,
            "system": system
        }

        if request.max_tokens:
            req["max_tokens"] = request.max_tokens
        if request.temperature:
            req["temperature"] = request.temperature
        if request.top_p:
            req["top_p"] = request.top_p
        if request.stop:
            req["stop_sequences"] = request.stop
        if request.tools:
            tools = self._convert_tools(request.tools)
            tool_choice = self._convert_tool_choice(tools is not None, request.tool_choice, request.parallel_tool_calls)
            req["tools"] = tools
            req["tool_choice"] = tool_choice

        anthropic_request = Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(**req)
        )
        return anthropic_request
    
    def _convert_response_message(self, message):
        if isinstance(message.content, str):
            return {
                "role": message.role,
                "content": message.content
            }
        elif isinstance(message.content, list):
            tool_calls = []
            content = []
            for item in message.content:
                if item.type == "tool_use":
                    tool_calls.append(
                        {
                            "type": "function",
                            "id": item.id,
                            "function":{
                                "name": item.name,
                                "arguments": item.input
                            }
                        }
                    )
                else:
                    content.append(item.to_dict())
            
            return {
                "role": message.role,
                "content": content,
                "tool_calls": tool_calls
            }

    def _convert_response(self, response) -> dict:
        if response.result.type == "succeeded":
            message = response.result.message

            choice = {
                "index": 0,
                "logprobs": None,
                "finish_reason": message.stop_reason.lower(),
                "message": self._convert_response_message(message)
            }
            choices = [choice]
            usage = {
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.input_tokens + message.usage.output_tokens
            }
            body = {
                "id": f'{response.custom_id}',
                "object": "chat.completion",
                "created": int(time.time()),
                "model": message.model,
                "system_fingerprint": None,
                "choices": choices,
                "usage": usage
            }
            res = {
                "request_id": response.custom_id,
                "status_code": 200,
                "body": body,
            }

            error = None
        elif response.result.type == "errored":
            error = {
                "message": response.result.error.type,
                "code": response.result.error.type
            }
            res = None
        elif response.result.type == "expired":
            error = {
                "message": "Request expired",
                "code": "request_expired"
            }
            res = None
            
        # create output
        output = {
            "id": f'{response.custom_id}',
            "custom_id": response.custom_id,
            "response": res,
            "error": error
        }
        return output

    def _validate_request(self, request):
        AnthropicChatCompletionRequest(**request)
