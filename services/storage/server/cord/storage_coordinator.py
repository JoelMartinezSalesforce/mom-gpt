from typing import Dict, List

import numpy as np
from icecream import ic
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType
from pymilvus.exceptions import CollectionNotExistException

from exceptions.connection.connection_exception import ConnectionException
from exceptions.storage.storage_exception import StorageException
from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.storage.server.server_status import ServerStorageStatus


class StorageWriter:
    """
    A non-thread-safe singleton class to ease implementing a coordinator to check the processes and state
    of the server meaning operations will be implemented here. and their state is according.
    """
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls, encoder: JSONEncoder):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            # Put any initialization here
            cls._instance._connect()
            cls.encoder = encoder
            cls._instance.status_cord = ServerStorageStatus.IS_AVAILABLE
        return cls._instance

    def __new__(cls, *args, **kwargs):
        return super(StorageWriter, cls).__new__(cls)

    def _connect(self):
        try:
            connections.connect(
                alias="default",
                user='username',
                password='password',
                host='localhost',
                port='19530'
            )
            self.is_connected = True
        except ConnectionException as e:
            self.is_connected = False
            ic(f"Milvus server is not active or not connected: {e}")

    def insert_data_and_create_collection(
            self,
            vectors: np.ndarray = None,
            index: Dict[str, any] = None,
            data: List[str] = None,
            collection_name: str = None,
            data_max_length: int = 512):

        # check important information should not be None
        if vectors is None:
            raise RuntimeError("Arg: 'vectors' cannot be None: Insertion of None object")

        if index is None:
            raise RuntimeError("Arg: 'index' cannot be None: Insertion of None object")

        if data is None:
            raise RuntimeError("Arg: 'data' cannot be None: Insertion of None object")

        if collection_name is None:
            raise RuntimeError("Arg: 'collection_name' cannot be None: Insertion of None object")

        embedding_dim = vectors.shape[1]

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=data_max_length),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        ])

        collection = Collection(name=collection_name, schema=schema)
        entities = [
            data,  # text
            vectors.tolist()  # vector
        ]

        try:

            insert_result = collection.insert(entities)
        except StorageException as e:
            raise StorageException(
                message="Insertion of data failed please check the Schema and corpus of the data, consistency can be a factor",
                errors=e
            )

        if index is None:
            index = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            }

        collection.create_index("embeddings", index)
        collection.load()

        return insert_result

    def insert_to_collection(self, collection_name: str = None, data: Dict[str, any] = None):
        def schema_is_similar(schema1: CollectionSchema, schema2: Dict[str, any], has_auto_id: bool = False) -> bool:

            if has_auto_id:

                data = [item for item in schema1.to_dict()['fields'] if not 'auto_id' in item]

            else:

                data = schema1.to_dict()['fields']

            # check if the schema has the same names as the schema such as {'': "", '': ""} and they should be the same


        if collection_name is None or data is None:
            raise RuntimeError("Collection name and data cannot be None.")

        try:
            collection = Collection(collection_name)
        except CollectionNotExistException:
            raise StorageException(f"Collection {collection_name} does not exist.")

        # Check if schemas are similar
        if not schema_is_similar(collection.schema, data, collection.schema['auto_id']):
            raise StorageException("Schema mismatch. Ensure that the data fields match the collection schema.")

        # Prepare data for upsert
        entities = []
        for field in collection.schema.fields:
            if field.name in data:
                entities.append(data[field.name])
            else:
                raise StorageException(f"Missing data for field '{field.name}'.")

        # Perform the upsert operation
        try:
            insert_result = collection.upsert(entities)
            print(f"Data upserted successfully: {insert_result}")
        except Exception as e:
            raise StorageException(f"Error during data upsert: {e}")

