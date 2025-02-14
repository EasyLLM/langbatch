import time
import pytest
import os

from openai import AzureOpenAI
from langbatch.openai import OpenAIChatCompletionBatch, OpenAIEmbeddingBatch
from tests.unit.fixtures import test_data_file, temp_dir, batch, embedding_batch

# @pytest.fixture
# def azure_openai_batch(test_data_file: str):
#     azure_endpoint = os.getenv("AZURE_API_BASE")
#     api_key = os.getenv("AZURE_API_KEY")
#     api_version = os.getenv("AZURE_API_VERSION")
#     client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)
#     batch = OpenAIChatCompletionBatch(test_data_file, client)
#     return batch

@pytest.mark.slow
def test_openai_batch_start(batch: OpenAIChatCompletionBatch):
    batch.start()
    assert batch.platform_batch_id is not None

    time.sleep(5)
    assert batch.get_status() == "in_progress"

# @pytest.mark.slow
# def test_azure_openai_batch_start(azure_openai_batch: OpenAIChatCompletionBatch):
#     azure_openai_batch.start()
#     assert azure_openai_batch.platform_batch_id is not None

#     time.sleep(5)
#     assert azure_openai_batch.get_status() == "in_progress"

@pytest.mark.slow
def test_openai_embedding_batch_start(embedding_batch: OpenAIEmbeddingBatch):
    embedding_batch.start()
    assert embedding_batch.platform_batch_id is not None

    time.sleep(5)
    assert embedding_batch.get_status() == "in_progress"