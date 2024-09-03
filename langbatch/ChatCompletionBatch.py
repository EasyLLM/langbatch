from typing import Iterable, List, Dict, Any, Tuple
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from langbatch.Batch import Batch

class ChatCompletionBatch(Batch):
    _url: str = "/v1/chat/completions"

    def __init__(self, file) -> None:
        super().__init__(file)

    @classmethod
    def create(cls, data: List[Iterable[ChatCompletionMessageParam]], **kwargs) -> "ChatCompletionBatch":
        """
        Create a chat completion batch when given a list of messages.

        Args:
            data (List[Iterable[ChatCompletionMessageParam]]): A list of messages to be sent to the API.
            **kwargs: Additional keyword arguments for the API call. Ex. model, messages, etc.

        Returns:
            ChatCompletionBatch: An instance of the ChatCompletionBatch class.

        Raises:
            ValueError: If the input data is invalid.

        Usage:
        ```python
        batch = ChatCompletionBatch.create([
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}],
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of Germany?"}]
            ],
            model="gpt-4o")
        ```
        """
        return cls._create_batch_file("messages", data, **kwargs)
        
    def get_results(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]] | Tuple[None, None]:
        """
        Retrieve the results of the chat completion batch.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]] | Tuple[None, None]: 
            A tuple containing successful and unsuccessful results.
            - Successful results: A list of dictionaries with "choices" and "custom_id" keys.
            - Unsuccessful results: A list of dictionaries with "error" and "custom_id" keys.

        Usage:
        ```python
        successful_results, unsuccessful_results = batch.get_results()
        for result in successful_results:
            print(result["choices"])
        ```
        """
        process_func = lambda result: {"choices": result['response']['body']['choices']}
        return self._prepare_results(process_func)
