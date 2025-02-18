from abc import abstractmethod
import logging
import json
from typing import Any, Dict

import jsonlines
from vertexai.preview.batch_prediction import BatchPredictionJob

from langbatch.Batch import Batch
from langbatch.ChatCompletionBatch import ChatCompletionBatch
from langbatch.bigquery_utils import write_data_to_bigquery, read_data_from_bigquery, create_table
from langbatch.schemas import VertexAIChatCompletionRequest, VertexAILlamaChatCompletionRequest, AnthropicChatCompletionRequest
from langbatch.claude_utils import convert_request, convert_message
from langbatch.errors import BatchStartError, BatchStateError

vertexai_state_map = {
    'JOB_STATE_UNSPECIFIED': 'unspecified',
    'JOB_STATE_QUEUED': 'in_progress',
    'JOB_STATE_PENDING': 'validating',
    'JOB_STATE_RUNNING': 'in_progress',
    'JOB_STATE_SUCCEEDED': 'completed',
    'JOB_STATE_FAILED': 'failed',
    'JOB_STATE_CANCELLING': 'cancelling',
    'JOB_STATE_CANCELLED': 'cancelled',
    'JOB_STATE_PAUSED': 'paused',
}

class VertexAIBatch(Batch):
    """
    VertexAIBatch is a class for Vertex AI batch processing.
    Implements the Batch class for Vertex AI API.
    """
    _url: str = "/v1/chat/completions"
    _field_name: str = "request"
    _publisher: str = "google"

    def __init__(self, file: str, model: str, gcp_project: str, bigquery_input_dataset: str, bigquery_output_dataset: str) -> None:
        """
        Initialize the VertexAIBatch class.

        Args:
            file (str): The path to the jsonl file in Vertex AI batch format.
            model (str): The name of the model to use for the batch prediction.
            gcp_project (str): The GCP project to use for the batch prediction.
            bigquery_input_dataset (str): The BigQuery dataset to use for the batch prediction input.
            bigquery_output_dataset (str): The BigQuery dataset to use for the batch prediction output.

        Usage:
        ```python
        batch = VertexAIBatch(
            "path/to/file.jsonl",
            "model",
            "gcp_project",
            "bigquery_input_dataset",
            "bigquery_output_dataset"
        )
        ```
        """
        super().__init__(file)

        self.model = model
        self.gcp_project = gcp_project
        self.bigquery_input_dataset = bigquery_input_dataset
        self.bigquery_output_dataset = bigquery_output_dataset

    @classmethod
    def _get_init_args(cls, meta_data) -> Dict[str, Any]:
        args = {
            "model": meta_data["model"],
            "gcp_project": meta_data["gcp_project"],
            "bigquery_input_dataset": meta_data["bigquery_input_dataset"],
            "bigquery_output_dataset": meta_data["bigquery_output_dataset"]
        }
        return args
    
    def _create_meta_data(self) -> Dict[str, Any]:
        meta_data = {
            "model": self.model,
            "gcp_project": self.gcp_project,
            "bigquery_input_dataset": self.bigquery_input_dataset,
            "bigquery_output_dataset": self.bigquery_output_dataset
        }

        return meta_data

    def _create_table(self, dataset_id: str):
        return create_table(self.gcp_project, dataset_id, self.id, self._field_name)

    @abstractmethod
    def _convert_request(self, req: dict) -> str:
        pass

    def _prepare_data(self):
        requests = self._get_requests()
        data = []
        for request in requests:
            data.append(self._convert_request(request))
        return data

    def _upload_batch_file(self):
        if self.platform_batch_id is None:
            self._create_table(self.bigquery_input_dataset)

        data = self._prepare_data()
        status = write_data_to_bigquery(self.gcp_project, self.bigquery_input_dataset, self.id, data, self._field_name)
        if not status:
            raise BatchStartError("Error writing data to BigQuery")
        
        return f"bq://{self.gcp_project}.{self.bigquery_input_dataset}.{self.id}"

    def _create_batch(self, input_dataset, output_dataset):
        job = BatchPredictionJob.submit(
            f"publishers/{self._publisher}/models/{self.model}",
            input_dataset,
            output_uri_prefix = output_dataset
        )

        self.platform_batch_id = job.name

    def start(self):
        if self.platform_batch_id is not None:
            raise BatchStateError("Batch already started")
        
        input_dataset = self._upload_batch_file()
        output_dataset_id = self._create_table(self.bigquery_output_dataset)
        output_dataset = f"bq://{self.gcp_project}.{self.bigquery_output_dataset}.{output_dataset_id}"
        self._create_batch(input_dataset, output_dataset)
    
    def get_status(self):
        if self.platform_batch_id is None:
            raise BatchStateError("Batch not started")
        
        job = BatchPredictionJob(self.platform_batch_id)
        return vertexai_state_map[str(job.state.name)]

    @abstractmethod
    def _convert_response(self, response):
        pass

    def _download_results_file(self):
        data = read_data_from_bigquery(self.gcp_project, self.bigquery_output_dataset, self.id)
        responses = []
        for element in data:
            response = self._convert_response(element)
            if response["response"] is not None:
                responses.append(response)

        file_path = self._create_results_file_path()
        with jsonlines.open(file_path, mode='w') as writer:
            writer.write_all(responses)

        return file_path

    def _get_errors(self):
        job = BatchPredictionJob(self.platform_batch_id)
        job_object = job.to_dict()
        if 'error' in job_object:
            return job_object['error']['message']
        else:
            return None
    
    def is_retryable_failure(self) -> bool:
        # TODO: implement retry logic for Vertex AI API
        error = self._get_errors()
        if error:
            logging.error(f"Error in VertexAI Batch: {error}")
            if "Failed to import data. Not found: Dataset" in error:
                return False
            else:
                return False
        else:
            return False

    def retry(self):
        if self.platform_batch_id is None:
            raise BatchStateError("Batch not started")
        
        job = BatchPredictionJob(self.platform_batch_id)
        input_dataset = job._gca_resource.input_config.bigquery_source.input_uri
        output_dataset = job._gca_resource.output_config.bigquery_destination.output_uri

        self._create_batch(input_dataset, output_dataset)

class VertexAIChatCompletionBatch(VertexAIBatch, ChatCompletionBatch):
    """
    VertexAIChatCompletionBatch is a class for Vertex AI chat completion batches.
    Can be used for batch processing with Gemini 1.5 Flash and Gemini 1.5 Pro models
    
    Usage:
    ```python
    batch = VertexAIChatCompletionBatch("path/to/file.jsonl")
    batch.start()
    ```
    """
    _url: str = "/v1/chat/completions"

    def _convert_request(self, req: dict) -> str:
        custom_schema = {
            "contents": [],
            "systemInstruction": None,
            "tools": [],
            "generationConfig": {}
        }
        request = json.loads(VertexAIChatCompletionRequest(**req["body"]).model_dump_json())

        # Track tool responses to match with tool calls
        tool_responses = {}
        
        # First pass - collect tool responses
        for message in request["messages"]:
            if message["role"] == "tool":
                tool_call_id = message["tool_call_id"]
                if tool_call_id:
                    tool_responses[tool_call_id] = {
                        "response": json.loads(message["content"])
                    }
        
        # Second pass - process messages
        for message in request["messages"]:
            role = message["role"]
            content = message["content"]

            function_calls = []
            tool_responses_cache = []
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    function_calls.append({
                        "functionCall": {
                            "name": tool_call["function"]["name"],
                            "args": json.loads(tool_call["function"]["arguments"])
                        }
                    })
                    # If we have a response for this tool call, add it in the next message
                    if tool_call["id"] in tool_responses:
                        response = tool_responses[tool_call["id"]]
                        response["name"] = tool_call["function"]["name"]
                        tool_responses_cache.append({
                            "functionResponse": response
                        })

            if role == "system":
                custom_schema["systemInstruction"] = {
                    "role": "system",
                    "parts": {"text": content}
                }
            elif role != "tool":  # Skip tool messages as we handle them separately
                if len(function_calls) > 0:
                    custom_schema["contents"].append({
                        "role": role,
                        "parts": function_calls
                    })
                    custom_schema["contents"].append({
                        "role": "model",
                        "parts": tool_responses_cache
                    })
                elif content is not None:
                    custom_schema["contents"].append({
                        "role": role,
                        "parts": {"text": content}
                    })

        # Convert tools
        if request["tools"]:
            for tool in request["tools"]:
                function = tool.get("function", {})
                
                custom_schema["tools"].append({
                    "functionDeclarations": [{
                        "name": function.get("name"),
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {})
                    }]
                })

        # Convert generation config
        gen_config = custom_schema["generationConfig"]
        if request.get("temperature"):
            gen_config["temperature"] = request["temperature"]
        if request.get("top_p"):
            gen_config["topP"] = request["top_p"]
        if request.get("max_tokens"):
            gen_config["maxOutputTokens"] = request["max_tokens"]
        if request.get("n"):
            gen_config["candidateCount"] = request["n"]
        if request.get("presence_penalty"):
            gen_config["presencePenalty"] = request["presence_penalty"]
        if request.get("frequency_penalty"):
            gen_config["frequencyPenalty"] = request["frequency_penalty"]
        if request.get("stop"):
            gen_config["stopSequences"] = request["stop"] if isinstance(request["stop"], list) else [request["stop"]] if request["stop"] else None
        if request.get("seed"):
            gen_config["seed"] = request["seed"]

        if request.get("response_format"):
            mime_type_map = {
                "json_object": "application/json",
                "text": "text/plain",
                "json_schema": "application/json"
            }

            gen_config["responseMimeType"] = mime_type_map[request["response_format"]["type"]]

            if request["response_format"]["type"] == "json_schema" and request["response_format"]["json_schema"]:
                gen_config["responseSchema"] = request["response_format"]["json_schema"]

                # Check for single enum property to use text/x.enum mime type
                data = json.loads(request["response_format"]["json_schema"]["schema"])
                concrete_types = ["string", "number", "integer", "boolean"]
                if data.get("type") in concrete_types and len(data.get("enum", [])) == 0:
                    gen_config["responseMimeType"] = "text/x.enum"


        gen_config = {k: v for k, v in gen_config.items() if v is not None}
        custom_schema["generationConfig"] = gen_config

        return { 
            "custom_id": req["custom_id"],
            "request": json.dumps(custom_schema, indent=2)
        }

    def _convert_response(self, response):
        # Parse the input JSON
        response_data = json.loads(response["response"])

        status = response["status"]
        if status != "":
            if "Bad Request: " in status:
                error_data = json.loads(status.split("Bad Request: ")[1])
            else:
                error_data = {
                    "message": status,
                    "code": "server_error"
                }

            error = {
                "message": error_data["error"]["message"],
                "code": error_data["error"]["code"]
            }

            res = None
        else:
            # Extract relevant information
            candidates = response_data["candidates"]
            tokens = response_data["usageMetadata"]

            # Create the choices array
            choices = []
            for index, candidate in enumerate(candidates):
                choice = {
                    "index": index,
                    "logprobs": None,
                    "finish_reason": candidate["finishReason"].lower()
                }

                tool_calls = []
                text_part = None
                for part in candidate["content"]["parts"]:
                    if part.get("functionCall", None):
                        tool_call = {
                            "type": "function",
                            "function": {
                                "name": part["functionCall"].get("name"),
                                "arguments": json.dumps(part["functionCall"].get("args", {}))
                            }
                        }
                        tool_calls.append(tool_call)
                    else:
                        text_part = part["text"]
                
                message = {
                    "role": "assistant",
                    "content": text_part
                }
                if len(tool_calls) > 0:
                    message["tool_calls"] = tool_calls

                choice["message"] = message
                choices.append(choice)

            usage = {
                "prompt_tokens": tokens.get("promptTokenCount", 0),
                "completion_tokens": tokens.get("candidatesTokenCount", 0),
                "total_tokens": tokens.get("totalTokenCount", 0)
            }

            # Create the body
            body = {
                "id": f'{response["custom_id"]}',
                "object": "chat.completion",
                "created": int(response["processed_time"].timestamp()),
                "model": self.model,
                "system_fingerprint": None,
                "choices": choices,
                "usage": usage
            }

            res = {
                "request_id": response["custom_id"],
                "status_code": 200,
                "body": body,
            }

            error = None

        # create output
        output = {
            "id": f'{response["custom_id"]}',
            "custom_id": response["custom_id"],
            "response": res,
            "error": error
        }

        return output

    def _validate_request(self, request):
        VertexAIChatCompletionRequest(**request)

class VertexAIClaudeChatCompletionBatch(VertexAIBatch, ChatCompletionBatch):
    _url: str = "/v1/chat/completions"
    _publisher: str = "anthropic"
    _field_name: str = "request"

    def _convert_request(self, req: dict) -> str:
        request = convert_request(req)
        request["anthropic_version"] = "vertex-2023-10-16"
        del request["model"]

        return {
            "custom_id": req["custom_id"],
            "request": json.dumps(request, indent=2)
        }

    def _convert_response(self, response):
        response_data = json.loads(response["response"])

        status = response["status"]
        if status != "":
            if "Bad Request: " in status:
                error_data = json.loads(status.split("Bad Request: ")[1])
            else:
                error_data = {
                    "message": status,
                    "code": "server_error"
                }

            error = {
                "message": error_data["error"]["message"],
                "code": error_data["error"]["code"]
            }

            res = None
        else:
            res= convert_message(response_data, response["custom_id"])
            error = None

        output = {
            "id": f'{response["custom_id"]}',
            "custom_id": response["custom_id"],
            "response": res,
            "error": error
        }

        return output

    def _validate_request(self, request):
        AnthropicChatCompletionRequest(**request)

class VertexAILlamaChatCompletionBatch(VertexAIBatch, ChatCompletionBatch):
    _url: str = "/v1/chat/completions"
    _publisher: str = "meta"
    _field_name: str = "body"

    def _convert_request(self, req: dict) -> str:
        request = VertexAILlamaChatCompletionRequest(**req["body"])

        request.model = f"meta/{self.model}"

        return {
            "custom_id": req["custom_id"],
            "body": request.model_dump_json()
        }

    def _convert_response(self, response):
        response_data = json.loads(response["response"])
        res = {
            "request_id": response["custom_id"],
            "status_code": 200,
            "body": response_data,
            
        }
        output = {
            "id": f'{response["id"]}',
            "custom_id": response["custom_id"],
            "response": res,
            "error": response["error"]
        }

        return output

    def _validate_request(self, request):
        VertexAILlamaChatCompletionRequest(**request)