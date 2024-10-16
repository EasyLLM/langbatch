import logging
from google.cloud import bigquery_storage_v1
from google.cloud.bigquery_storage_v1 import types, writer
from google.protobuf import descriptor_pb2
from google.cloud import bigquery
from langbatch.record_pb2 import BatchRecord
import time

def create_row_data(custom_id: str, request: str):
    row = BatchRecord()
    row.custom_id = custom_id
    row.request = request
    return row.SerializeToString()

def write_data_to_bigquery(project_id: str, dataset_id: str, table_id: str, data: list):
    try:
        write_client = bigquery_storage_v1.BigQueryWriteClient()
        parent = write_client.table_path(project_id, dataset_id, table_id)
    
        write_stream = types.WriteStream()
        write_stream.type_ = types.WriteStream.Type.COMMITTED
        write_stream = write_client.create_write_stream(parent=parent, write_stream=write_stream)
        stream_name = write_stream.name
    except:
        logging.error("Error creating write stream", exc_info=True)
        raise ValueError("Error writing data to BigQuery. Check the GCP project and BigQuery dataset values")

    try:
        # Create a template with fields needed for the first request.
        request_template = types.AppendRowsRequest()

        # The initial request must contain the stream name.
        request_template.write_stream = stream_name

        # So that BigQuery knows how to parse the serialized_rows, generate a
        # protocol buffer representation of your message descriptor.
        proto_schema = types.ProtoSchema()
        proto_descriptor = descriptor_pb2.DescriptorProto()
        proto_descriptor.name = "Row"
        proto_descriptor.field.add(name="custom_id", number=1, type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        proto_descriptor.field.add(name="request", number=2, type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        proto_schema.proto_descriptor = proto_descriptor
        proto_data = types.AppendRowsRequest.ProtoData()
        proto_data.writer_schema = proto_schema
        request_template.proto_rows = proto_data

        # Create an AppendRowsStream.
        append_rows_stream = writer.AppendRowsStream(write_client, request_template)

        # Write data in batches
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            proto_rows = types.ProtoRows()
            for item in batch:
                proto_rows.serialized_rows.append(create_row_data(item['custom_id'], item['request']))

            request = types.AppendRowsRequest()
            request.offset = i
            proto_data = types.AppendRowsRequest.ProtoData()
            proto_data.rows = proto_rows
            request.proto_rows = proto_data

            append_rows_stream.send(request)

        append_rows_stream.close()

        return True
    except:
        logging.error("Error writing data to BigQuery", exc_info=True)
        return False

def create_table(project_id: str, dataset_id: str, id: str):
    client = bigquery.Client()
    schema = [
        bigquery.SchemaField("request", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("custom_id", "STRING", mode="REQUIRED"),
    ]

    table_id = f"{project_id}.{dataset_id}.{id}"
    table = bigquery.Table(table_id, schema=schema)
    
    try:
        table = client.create_table(table)
    except Exception as e:
        if "Already Exists:" in str(e):
            drop_table(project_id, dataset_id, id)
            table = client.create_table(table)
        else:
            raise e
        
    time.sleep(10)
        
    return table.table_id

def drop_table(project_id: str, dataset_id: str, table_id: str):
    client = bigquery.Client()
    table_id = f"{project_id}.{dataset_id}.{table_id}"
    client.delete_table(table_id, not_found_ok=True)

def read_data_from_bigquery(project_id: str, dataset_id: str, table_id: str):
    client = bigquery_storage_v1.BigQueryReadClient()

    # Initialize table
    table = f"projects/{project_id}/datasets/{dataset_id}/tables/{table_id}"

    # Initialize parent
    parent = f'projects/{project_id}'

    requested_session = bigquery_storage_v1.types.ReadSession(
        table=table,
        data_format=bigquery_storage_v1.types.DataFormat.AVRO,
    )
    session = client.create_read_session(
        parent=parent, read_session=requested_session
    )

    stream = session.streams[0]  # Read the first stream.
    read_rows_stream = client.read_rows(stream.name)

    data = []
    for element in read_rows_stream.rows(session):
        data.append(element)

    return data