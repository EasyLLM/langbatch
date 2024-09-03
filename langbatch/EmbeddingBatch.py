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
        file_path = cls._create_batch_file("input", data, **kwargs)
        return cls(file_path)
    
    def get_results(self):
        process_func = lambda result: {"embedding": result['response']['body']['data'][0]['embedding']}
        return self._prepare_results(process_func)
                