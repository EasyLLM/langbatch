# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: record_llama.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12record_llama.proto\x12\x0crecord_llama\"K\n\x0b\x42\x61tchRecord\x12\x11\n\tcustom_id\x18\x01 \x02(\t\x12\x0e\n\x06method\x18\x02 \x02(\t\x12\x0b\n\x03url\x18\x03 \x02(\t\x12\x0c\n\x04\x62ody\x18\x04 \x02(\t')



_BATCHRECORD = DESCRIPTOR.message_types_by_name['BatchRecord']
BatchRecord = _reflection.GeneratedProtocolMessageType('BatchRecord', (_message.Message,), {
  'DESCRIPTOR' : _BATCHRECORD,
  '__module__' : 'record_llama_pb2'
  # @@protoc_insertion_point(class_scope:record_llama.BatchRecord)
  })
_sym_db.RegisterMessage(BatchRecord)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _BATCHRECORD._serialized_start=36
  _BATCHRECORD._serialized_end=111
# @@protoc_insertion_point(module_scope)
