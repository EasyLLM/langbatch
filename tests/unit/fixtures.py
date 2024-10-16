import tempfile
import shutil
from pathlib import Path

import pytest

from langbatch.openai import OpenAIChatCompletionBatch
from langbatch.openai import OpenAIEmbeddingBatch

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

def copy_file(file_name, temp_dir):
    source_path = Path('tests/data') / file_name
    dest_path = Path(temp_dir) / file_name
    shutil.copy(source_path, dest_path)
    return dest_path

@pytest.fixture
def test_data_file(temp_dir, request):
    file_name = request.param if hasattr(request, 'param') else 'chat_completion_batch.jsonl'
    return copy_file(file_name, temp_dir)

@pytest.fixture
def batch(temp_dir) -> OpenAIChatCompletionBatch:
    file_name = 'chat_completion_batch.jsonl'
    dest_path = copy_file(file_name, temp_dir)
    return OpenAIChatCompletionBatch(dest_path)

@pytest.fixture
def embedding_batch(temp_dir) -> OpenAIEmbeddingBatch:
    file_name = 'embedding_batch.jsonl'
    dest_path = copy_file(file_name, temp_dir)
    return OpenAIEmbeddingBatch(dest_path)