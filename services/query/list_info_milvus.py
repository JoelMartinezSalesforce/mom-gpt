from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

if __name__ == '__main__':

    # Initialize connection to Milvus
    connections.connect("default", host='localhost', port='19530', user='username', password='password')

    # Define the schema for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)  # Assuming dimension is 128
    ]
    schema = CollectionSchema(fields=fields)

    # Create or get the collection
    collection_name = "simple_test_collection"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    collection = Collection(name=collection_name, schema=schema)

    # Insert sample data
    entities = [
        ["test data"],  # Data for VARCHAR field
        [[0.1] * 128]  # Data for FLOAT_VECTOR field
    ]
    insert_result = collection.insert(entities)
    print("Insertion result:", insert_result)

    # Create an index on the 'embeddings' field
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 100}
    }
    collection.create_index("embeddings", index_params)
    print("Index created on 'embeddings' field.")

    # Load the collection
    collection.load()
    print("Collection is loaded and ready for search.")

