from typing import List, Dict, Any, Tuple
from langbatch.Batch import Batch

class EmbeddingBatch(Batch):
    """
    EmbeddingBatch is a base class for embedding batch classes.
    Utilizes OpenAI Embedding API format as the standard request format.
    """
    _url: str = "/v1/embeddings" 

    def __init__(self, file) -> None:
        """
        Initialize the EmbeddingBatch class.
        """
        super().__init__(file)

    @classmethod
    def create(cls, data: List[str], request_kwargs: Dict = {}, batch_kwargs: Dict = {}) -> "EmbeddingBatch":
        """
        Create an embedding batch when given a list of texts.

        Args:
            data (List[str]): A list of texts to be embedded.
            request_kwargs (Dict): Additional keyword arguments for the API call. Ex. model, encoding_format, etc.
            batch_kwargs (Dict): Additional keyword arguments for the batch class.

        Returns:
            An instance of the EmbeddingBatch class.

        Raises:
            ValueError: If the input data is invalid.

        Usage:
        ```python
        batch = OpenAIEmbeddingBatch.create([
            "Hello world", 
            "Hello LangBatch"
        ], 
            request_kwargs={"model": "text-embedding-3-small"})
        ```
        """
        return cls._create_batch_file("input", data, request_kwargs, batch_kwargs)
    
    def get_results(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]] | Tuple[None, None]:
        """
        Retrieve the results of the embedding batch.

        Returns:
            A tuple containing successful and unsuccessful results. Successful results: A list of dictionaries with "embedding" and "custom_id" keys. Unsuccessful results: A list of dictionaries with "error" and "custom_id" keys.

        Usage:
        ```python
        successful_results, unsuccessful_results = batch.get_results()
        for result in successful_results:
            print(result["embedding"])
        ```
        """
        process_func = lambda result: {"embedding": result['response']['body']['data'][0]['embedding']}
        return self._prepare_results(process_func)