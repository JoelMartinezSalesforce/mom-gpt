from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility, AnnSearchRequest, RRFRanker, MilvusClient, MilvusException
)
from pymilvus.client.types import MetricType


class MilvusStorageCoord:
    _instance = None
    _host = "localhost"
    _port = "19530"
    _client = MilvusClient(
        host=_host,
        port=_port
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MilvusStorageCoord, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """
        Initialize connection to Milvus.
        """
        print("Initializing Milvus Storage Coordination")
        connections.connect("default", host=self._host, port=self._port)
        print("Connected to Milvus")

    def create_and_index_collection(self, collection_name, fields, index_fields, index_params):
        """
        Create a collection with specified fields and indexes if it doesn't already exist.
        Args:
            collection_name (str): The name of the collection to create.
            fields (list): A list of dictionaries specifying field schemas.
            index_fields (list): A list of field names on which to create indexes.
            index_params (dict): Parameters for index creation.
        """
        if not utility.has_collection(collection_name):
            # Convert the list of field definitions into FieldSchema objects
            field_schemas = [FieldSchema(name=field['name'], dtype=field['type'], dim=field.get('dim', None), is_primary=field.get('is_primary', False)) for field in fields]
            schema = CollectionSchema(fields=field_schemas, description=f"Generic collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            print(f"Collection '{collection_name}' created.")

            # Create indexes for specified fields
            for field_name in index_fields:
                collection.create_index(field_name, index_params)
                print(f"Index created for field '{field_name}' with parameters {index_params}.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    def hybrid_search(self, collection_name, first_query_vector: list = None, second_query_vector: list = None,
                      limit: int = 5):
        """
        Perform a hybrid search using multiple vector fields.
        """
        if second_query_vector is None:
            second_query_vector = []
        if first_query_vector is None:
            first_query_vector = []
        collection = Collection(name=collection_name)
        collection.load()

        # Define ANN search requests
        request_1 = AnnSearchRequest(
            data=[first_query_vector],
            anns_field="filmVector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit
        )

        request_2 = AnnSearchRequest(
            data=[second_query_vector],
            anns_field="posterVector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit
        )

        # Use RRFRanker for reciprocal rank fusion reranking
        rerank = RRFRanker()

        # Perform hybrid search
        res = collection.hybrid_search(
            reqs=[request_1, request_2],
            rerank=rerank,
            limit=limit
        )

        return res

    def single_vector_query(self, collection_name, query_vector=None, limit=5):
        if query_vector is None:
            raise ValueError("Query vector is not provided.")

        # Ensure the query vector is in the right format (list of lists for Milvus 2.x)
        if not isinstance(query_vector[0], list):
            query_vector = [query_vector]

        # Define the search parameters
        search_params = {
            "metric_type": MetricType.L2,
            "params": {"nprobe": 10}
        }

        try:
            results = self._client.search(
                collection_name=collection_name,  # Specify the collection name
                data=query_vector,                # Query vectors
                anns_field='embedding',           # Field in the collection to search against
                param=search_params,              # Search parameters
                limit=limit,                      # Number of results to return
                output_fields=['data']            # Fields to include in the result
            )
            return results
        except MilvusException as e:
            return f"Search failed due to {e.message}"


if __name__ == "__main__":
    # Example usage
    coordinator = MilvusStorageCoord()
    fields = [
        {'name': 'item_id', 'type': DataType.INT64, 'is_primary': True},
        {'name': 'itemVector', 'type': DataType.FLOAT_VECTOR, 'dim': 128},
        {'name': 'tagVector', 'type': DataType.FLOAT_VECTOR, 'dim': 128}
    ]
    index_fields = ['itemVector', 'tagVector']
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    coordinator.create_and_index_collection("my_custom_collection", fields, index_fields, index_params)
