import uuid
from typing import List, Union, Iterable
from typing_extensions import Literal, Required, TypedDict
import jsonlines
from openai import OpenAI
from openai.types import EmbeddingCreateParams
from langbatch.schemas import OpenAIChatCompletionRequest
from langbatch.Batch import Batch

class EmbeddingBatch(Batch):
    _url: str = "/v1/embeddings"

    def __init__(self, file):
        super().__init__(file)

    @classmethod
    def create(cls, data: List[str], **kwargs):
        requests = []
        for input in data:
            try:
                body = kwargs
                body["input"] = input
                request = {
                    "custom_id": str(uuid.uuid4()),
                    "method": "POST",
                    "url": cls._url,
                    "body": body
                }
                requests.append(request)
            except:
                continue

        file_path = cls._prepare_batch(requests)
        return cls(file_path)

    def get_results(self):
        file_id = self._download_results_file()

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
                    "embedding": result['response']['body']['data'][0]['embedding']
                    }
                successful_results.append(choices)
            else:
                error = {
                    "custom_id": result['custom_id'],
                    "error": result['error']
                }
                unsuccessful_results.append(error)

        return successful_results, unsuccessful_results
    
class OpenAIEmbeddingBatch(EmbeddingBatch):
    def __init__(self, file, client: OpenAI = OpenAI()):
        super().__init__(file)
        self.client = client

    def _upload_batch_file(self):
        # Upload the batch file to OpenAI
        with open(self._file, "rb") as file:
            batch_input_file  = self.client.files.create(file=file, purpose="batch")
            return batch_input_file.id
        
    def start(self):
        batch_input_file_id = self._upload_batch_file()
        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=self._url,
            completion_window= "24h"
        )
        self.openai_batch_id = batch.id
    
    def cancel(self):
        batch = self.client.batches.cancel(self.openai_batch_id)
        if batch.status == "cancelling" or batch.status == "cancelled":
            return True
        else:
            return False
        
    def status(self):
        batch = self.client.batches.retrieve(self.openai_batch_id)
        return batch.status
    
    def _validate_request(self, request):
        OpenAIEmbeddingRequest(**request)

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
                