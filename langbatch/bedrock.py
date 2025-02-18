from typing import Any, Dict
from pathlib import Path
import jsonlines
import boto3
import botocore

from langbatch.Batch import Batch
from langbatch.ChatCompletionBatch import ChatCompletionBatch
from langbatch.schemas import AnthropicChatCompletionRequest, OpenAIChatCompletionRequest
from langbatch.nova_utils import convert_response_nova, convert_request_nova
from langbatch.claude_utils import convert_request, convert_message
from langbatch.errors import BatchStateError, BatchResultsError

bedrock_state_map = {
    'Scheduled': 'in_progress',
    'InProgress': 'in_progress',
    'Submitted': 'in_progress',
    'Validating': 'in_progress',
    'Stopping': 'in_progress',
    'PartiallyCompleted': 'completed',
    'Completed': 'completed',
    'Failed': 'failed',
    'Stopped': 'cancelled',
    'Expired': 'expired',
}

class BedrockBatch(Batch):
    """
    BedrockBatch is a class for Bedrock batch processing.
    Implements the Batch class for Bedrock API.
    """
    _url: str = ""

    def __init__(self, file: str, model: str, input_bucket: str, output_bucket: str, region: str, service_role: str) -> None:
        """
        Initialize the BedrockBatch class.

        Args:
            file (str): The path to the jsonl file in OpenAI batch format.

        Usage:
        ```python
        batch = BedrockChatCompletionBatch(
            "path/to/file.jsonl",
            "model",
            "input_bucket",
            "output_bucket",
            "region"
        )
        ```
        """
        super().__init__(file)
        self._client = boto3.client(service_name='bedrock', region_name=region)
        self._s3_client = boto3.resource('s3')

        self.model = model
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.service_role = service_role
        self.region = region

    @classmethod
    def _get_init_args(cls, meta_data) -> Dict[str, Any]:
        args = {
            "model": meta_data["model"],
            "input_bucket": meta_data["input_bucket"],
            "output_bucket": meta_data["output_bucket"],
            "region": meta_data["region"],
            "service_role": meta_data["service_role"]
        }
        return args
    
    def _create_meta_data(self) -> Dict[str, Any]:
        meta_data = {
            "model": self.model,
            "input_bucket": self.input_bucket,
            "output_bucket": self.output_bucket,
            "region": self.region,
            "service_role": self.service_role
        }

        return meta_data

    def _prepare_data(self):
        requests = self._get_requests()
        return [self._convert_request(request) for request in requests]
    
    def _upload_batch_file(self):
        data = self._prepare_data()

        file_path = Path(f"{self.id}_prepared_data.jsonl")
        with jsonlines.open(file_path, mode='w') as writer:
            writer.write_all(data)

        self._s3_client.Bucket(self.input_bucket).upload_file(
            str(file_path),
            f'{self.id}/input.jsonl'
        )

        file_path.unlink(missing_ok=True)
    
    def _create_batch(self):
        self._upload_batch_file()
        job = self._client.create_model_invocation_job(
            roleArn = self.service_role,
            jobName = self.id,
            modelId = self.model,
            inputDataConfig = {"s3InputDataConfig": { "s3Uri":f's3://{self.input_bucket}/{self.id}/'}},
            outputDataConfig = {"s3OutputDataConfig": { "s3Uri":f's3://{self.output_bucket}/{self.id}/'}},
        )
        self.platform_batch_id = job['jobArn']

    def start(self):
        if self.platform_batch_id is not None:
            raise BatchStateError("Batch already started")
        
        self._create_batch()
    
    def get_status(self):
        if self.platform_batch_id is None:
            raise BatchStateError("Batch not started")
        
        job = self._client.get_model_invocation_job(
            jobIdentifier=self.platform_batch_id
        )
        return bedrock_state_map[job['status']]

    def _download_results_file(self):
        if self.platform_batch_id is None:
            raise BatchStateError("Batch not started")
        
        job_id = self.platform_batch_id.split("/")[-1]
        s3_path = f"{self.id}/{job_id}/input.jsonl.out"

        try:
            self._s3_client.Bucket(self.output_bucket).download_file(s3_path, f"{job_id}_results.jsonl")
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return None
            raise BatchResultsError("Failed to download results file from S3")
        
        data = []
        file_path = Path(f"{job_id}_results.jsonl")
        with jsonlines.open(file_path, mode='r') as reader:
            for line in reader:
                data.append(line)
        file_path.unlink(missing_ok=True)

        file_path = self._create_results_file_path()
        with jsonlines.open(file_path, mode='w') as writer:
            for result in data:
                writer.write(self._convert_response(result))

        return file_path

    def _get_errors(self):
        # Implement error retrieval logic for Anthropic API
        job = self._client.get_model_invocation_job(jobIdentifier=self.platform_batch_id)
        return job.get('message')
    
    def is_retryable_failure(self) -> bool:
        error = self._get_errors()
        if error:
            return False
        else:
            return False

    def retry(self):
        if self.platform_batch_id is None:
            raise BatchStateError("Batch not started")
        
        self._create_batch()

class BedrockNovaChatCompletionBatch(BedrockBatch, ChatCompletionBatch):
    """
    BedrockNovaChatCompletionBatch is a class for Bedrock chat completion batches with Nova models.
    
    Usage:
    ```python
    batch = BedrockNovaChatCompletionBatch(
        "path/to/file.jsonl", 
        "model",
        "input_bucket",
        "output_bucket",
        "region",
        "service_role"
    )
    batch.start()
    ```
    """
    def _convert_request(self, req: dict):
        custom_id = req["custom_id"]
        request = convert_request_nova(req)

        return {"recordId": custom_id, "modelInput": request}
    
    def _convert_response(self, response) -> dict:
        return convert_response_nova(response, self.model)

    def _validate_request(self, request):
        AnthropicChatCompletionRequest(**request)

class BedrockClaudeChatCompletionBatch(BedrockBatch, ChatCompletionBatch):
    """
    BedrockClaudeChatCompletionBatch is a class for Bedrock chat completion batches with Claude models.
    
    Usage:
    ```python
    batch = BedrockClaudeChatCompletionBatch(
        "path/to/file.jsonl", 
        "model",
        "input_bucket",
        "output_bucket",
        "region",
        "service_role"
    )
    batch.start()
    ```
    """
    def _convert_request(self, req: dict):
        custom_id = req["custom_id"]
        request = convert_request(req)
        request["anthropic_version"] = "bedrock-2023-05-31"
        del request["model"]

        return {"recordId": custom_id, "modelInput": request}
    
    def _convert_response(self, response) -> dict:
        res= convert_message(response['modelOutput'], response["recordId"])
        error = None

        output = {
            "id": f'{response["recordId"]}',
            "custom_id": response["recordId"],
            "response": res,
            "error": error
        }
        return output

    def _validate_request(self, request):
        AnthropicChatCompletionRequest(**request)
