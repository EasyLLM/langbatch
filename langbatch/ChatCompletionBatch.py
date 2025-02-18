from typing import Iterable, List, Dict, Any, Tuple
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from langbatch.Batch import Batch

class ChatCompletionBatch(Batch):
    """
    ChatCompletionBatch is a base class for chat completion batch classes.
    Utilizes OpenAI Chat Completion API format as the standard request format.
    """
    _url: str = "/v1/chat/completions"

    def __init__(self, file) -> None:
        """
        Initialize the ChatCompletionBatch class.
        """
        super().__init__(file)

    @classmethod
    def create(cls, data: List[Iterable[ChatCompletionMessageParam]], request_kwargs: Dict = {}, batch_kwargs: Dict = {}) -> "ChatCompletionBatch":
        """
        Create a chat completion batch when given a list of messages.

        Args:
            data (List[Iterable[ChatCompletionMessageParam]]): A list of messages to be sent to the API.
            request_kwargs (Dict): Additional keyword arguments for the API call. Ex. model, messages, etc.
            batch_kwargs (Dict): Additional keyword arguments for the batch class. Ex. gcp_project, etc. for VertexAIChatCompletionBatch.

        Returns:
            An instance of the ChatCompletionBatch class.

        Raises:
            BatchInitializationError: If the input data is invalid.

        Usage:
        ```python
        batch = OpenAIChatCompletionBatch.create([
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}],
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of Germany?"}]
            ],
            request_kwargs={"model": "gpt-4o"})

        # For Vertex AI
        batch = VertexAIChatCompletionBatch.create([
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}],
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of Germany?"}]
            ],
            request_kwargs={"model": "gemini-2.0-flash-001"},
            batch_kwargs={
                "gcp_project": "your-gcp-project", 
                "bigquery_input_dataset": "your-bigquery-input-dataset", 
                "bigquery_output_dataset": "your-bigquery-output-dataset"
            })
        ```
        """
        return cls._create_batch_file("messages", data, request_kwargs, batch_kwargs)
        
    def get_results(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]] | Tuple[None, None]:
        """
        Retrieve the results of the chat completion batch.

        Returns:
            A tuple containing successful and unsuccessful results. Successful results: A list of dictionaries with "choices" and "custom_id" keys. Unsuccessful results: A list of dictionaries with "error" and "custom_id" keys.

        Usage:
        ```python
        successful_results, unsuccessful_results = batch.get_results()
        for result in successful_results:
            print(result["choices"])
        ```
        """
        process_func = lambda result: {"choices": result['response']['body']['choices']}
        return self._prepare_results(process_func)