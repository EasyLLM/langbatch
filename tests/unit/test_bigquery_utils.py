import os
import json
import uuid
import pytest
import jsonlines
from google.cloud import bigquery

from langbatch.bigquery_utils import (
    write_data_to_bigquery,
    create_table,
    drop_table,
    read_data_from_bigquery
)
from tests.unit.fixtures import test_data_file, temp_dir

# Constants for BigQuery interaction
PROJECT_ID = os.environ["GCP_PROJECT"]
DATASET_ID = os.environ["BIGQUERY_INPUT_DATASET"]
TABLE_ID = 'test_table'

@pytest.fixture
def requests(test_data_file):
    requests = []
    with jsonlines.open(test_data_file) as reader:
        for req in reader:
            requests.append({"custom_id": req["custom_id"], "request": json.dumps(req["request"])})
    return requests

def test_create_and_drop_table():
    client = bigquery.Client()

    # Create a temporary table
    table_id = create_table(PROJECT_ID, DATASET_ID, TABLE_ID)
    assert table_id is not None

    # Drop the table
    drop_table(PROJECT_ID, DATASET_ID, TABLE_ID)

    # Check if the table was dropped
    with pytest.raises(Exception):
        client.get_table(table_id)

@pytest.mark.parametrize('test_data_file', ['vertexai_requests.jsonl'], indirect=True)
def test_write_and_read_data_from_bigquery(requests):
    table_id = str(uuid.uuid4())

    # Ensure the table doesn't exist
    drop_table(PROJECT_ID, DATASET_ID, table_id)

    try:
        # create the table
        table_id = create_table(PROJECT_ID, DATASET_ID, table_id)

        # Write data to BigQuery
        result = write_data_to_bigquery(PROJECT_ID, DATASET_ID, table_id, requests)
        assert result == True

        # Read data from BigQuery
        read_data = read_data_from_bigquery(PROJECT_ID, DATASET_ID, table_id)

        # Compare written and read data
        assert len(read_data) == len(requests)
        for written, read in zip(requests, read_data):
            assert written['custom_id'] == read['custom_id']
            assert written['request'] == read['request']
    finally:
        # Clean up: drop the table after the test
        drop_table(PROJECT_ID, DATASET_ID, table_id)

def test_write_data_to_bigquery_error():
    data = [
        {"custom_id": "1", "request": {"contents": [{"role": "user", "parts": {"text": "Give me a recipe for banana bread."}}]}},
        {"custom_id": "2", "request": {"contents": [{"role": "user", "parts": {"text": "Give me a recipe for chocolate cake."}}]}},
    ]
    with pytest.raises(ValueError, match="Error writing data to BigQuery. Check the GCP project and BigQuery dataset values"):
        write_data_to_bigquery("invalid-project-id", DATASET_ID, TABLE_ID, data)

    with pytest.raises(ValueError, match="Error writing data to BigQuery. Check the GCP project and BigQuery dataset values"):
        write_data_to_bigquery(PROJECT_ID, "invalid-dataset", TABLE_ID, data)

    with pytest.raises(ValueError, match="Error writing data to BigQuery. Check the GCP project and BigQuery dataset values"):
        write_data_to_bigquery(PROJECT_ID, DATASET_ID, "invalid-table", data)

    with pytest.raises(ValueError, match="Error writing data to BigQuery"):
        write_data_to_bigquery(PROJECT_ID, DATASET_ID, TABLE_ID, data)