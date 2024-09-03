from langbatch.OpenAIBatch import OpenAIBatch
from langbatch.ChatCompletionBatch import ChatCompletionBatch
from langbatch.EmbeddingBatch import EmbeddingBatch
from langbatch.schemas import OpenAIChatCompletionRequest, OpenAIEmbeddingRequest

class OpenAIChatCompletionBatch(OpenAIBatch, ChatCompletionBatch):
    _url: str = "/v1/chat/completions"

    def _validate_request(self, request):
        OpenAIChatCompletionRequest(**request)

class OpenAIEmbeddingBatch(OpenAIBatch, EmbeddingBatch):
    def _validate_request(self, request):
        OpenAIEmbeddingRequest(**request)