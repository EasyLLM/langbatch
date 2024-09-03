import uuid
from typing import Iterable, List
import jsonlines
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from langbatch.Batch import Batch

class ChatCompletionBatch(Batch):
    _url: str = "/v1/chat/completions"

    def __init__(self, file):
        super().__init__(file)

    @classmethod
    def create(cls, data: List[Iterable[ChatCompletionMessageParam]], **kwargs):
        file_path = cls._create_batch_file("messages", data, **kwargs)
        return cls(file_path)

    def get_results(self):
        process_func = lambda result: {"choices": result['response']['body']['choices']}
        return self._prepare_results(process_func)
