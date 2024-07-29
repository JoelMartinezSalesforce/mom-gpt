import csv
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from icecream import ic

# Encoder initialization
from services.model.embeddings.corpus.json_encoder import JSONEncoder

if __name__ == '__main__':
    # Connect to Milvus
    connections.connect("default", host='localhost', port='19530', user='username', password='password')
    ic("Collections in the system:", utility.list_collections())

    # Drop existing collections if they exist
    if utility.has_collection("health_data"):
        utility.drop_collection("health_data")

    # Initialize encoder
    encoder = JSONEncoder(
        json_file_path="/Users/isaacpadilla/milvus-dir/mom-gpt/services/model/embeddings/bare_testing/dump/network_health_cons.json"
    )

    # Preprocess data and encode
    preprocessed_data = encoder.preprocess_for_encoding()
    vector_res = encoder.model_wrapper.encode(texts=preprocessed_data[:3], flat=True)
    ic("Vector Results first sample:", str(vector_res[0][:10]) + "...")
    ic("Length of embeddings list: ", len(vector_res))
    for i, elem in enumerate(vector_res):
        ic(f"Length of individual embedding {i}: {len(elem)}")

    COLLECTION_NAME = "health_data"

    # Checking dimensions consistency
    # Example of verifying and printing dimensions
    expected_dim = len(vector_res[0])  # Assuming the first embedding is correctly sized
    print("Expected dimension per embedding:", expected_dim)

    all_dims_correct = all(len(emb) == expected_dim for emb in vector_res)
    if not all_dims_correct:
        print("Error: Not all embeddings have the same dimension")
    else:
        print("All embeddings have the correct dimension:", expected_dim)

    # Define or redefine the schema if necessary
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=expected_dim),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Prepare and insert entities
    entities = [
        preprocessed_data,  # List of strings for the "data" field
        vector_res  # List of embeddings, ensure this is a list of list of floats
    ]

    try:
        insert_result = collection.insert(entities)
        print("Insertion result:", insert_result)
    except Exception as e:
        print("Error during insertion:", str(e))

    # Create index and load collection
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 256}
    }
    collection.create_index("embeddings", index_params)
    ic("Index created on 'embeddings' field.")

    collection.load()
    ic("Collection is loaded and ready for search.")
