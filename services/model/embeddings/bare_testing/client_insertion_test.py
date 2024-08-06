from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from icecream import ic

# Assuming JSONEncoder is implemented as expected
from services.model.embeddings.corpus.json_encoder import JSONEncoder

if __name__ == '__main__':
    # Connect to Milvus
    connections.connect("default", host='localhost', port='19530', user='username', password='password')
    ic("Collections in the system:", utility.list_collections())

    COLLECTION_NAME = "health_data"
    # Drop existing collection if it exists
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        ic(f"Dropped existing collection: {COLLECTION_NAME}")

    # Initialize encoder
    encoder = JSONEncoder(
        json_file_path="path/to/object.json",
    )

    # Preprocess data and encode
    preprocessed_data = encoder.preprocess_for_encoding()
    vector_res = encoder.model_wrapper.encode(texts=preprocessed_data[:50], flat=True)
    ic("Vector Results first sample:", str(vector_res[0][:10]) + "...")
    ic("Length of embeddings list: ", len(vector_res))

    # Define collection schema in Milvus
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=len(vector_res[0])),
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=1024)  # Assuming color is a simple attribute
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    ic(f"Collection '{COLLECTION_NAME}' created with schema.")

    # Prepare data for insertion
    data = [{"vector": vector_res[i], "data": preprocessed_data[i]} for i in range(len(vector_res))]

    try:
        insert_result = collection.insert(data)
        ic("Insertion result:", insert_result)
    except Exception as e:
        ic("Error during insertion:", str(e))

    # Create index and load collection
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 256}
    }
    collection.create_index("vector", index_params)
    ic("Index created on 'vector' field.")

    collection.load()
    ic("Collection is loaded and ready for search.")
