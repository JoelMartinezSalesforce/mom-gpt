import csv
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

# Encoder initialization
from services.model.embeddings.corpus.json_encoder import JSONEncoder

if __name__ == '__main__':
    # Connect to Milvus
    connections.connect("default", host='localhost', port='19530', user='username', password='password')
    print("Collections in the system:", utility.list_collections())

    # Drop existing collections
    for collection_name in utility.list_collections():
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

    print("Collections in the system after global drop:", utility.list_collections())

    # Initialize encoder
    encoder = JSONEncoder(json_file_path="/Users/isaacpadilla/milvus-dir/mom-gpt/services/models/data/dump/data_dump.json")

    # Preprocess data and encode
    preprocessed_data = encoder.preprocess_for_encoding()
    vector_res = encoder.model_wrapper.encode(preprocessed_data)
    print("Vector Results first sample:", str(vector_res[0])[:150] + "...")

    # Save embeddings to a CSV file
    csv_file_path = 'embeddings.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Text", "Embedding"])
        for idx, (text, embedding) in enumerate(zip(preprocessed_data, vector_res)):
            writer.writerow([idx, text, embedding])

    print(f"Embeddings saved to {csv_file_path}")

    # Define collection schema in Milvus
    COLLECTION_NAME = "health_data"
    embedding_dim = encoder.model_wrapper.encoding_dimensions
    print("Embedding Dimension:", embedding_dim)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=2)
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Prepare entities correctly
    entities = [
        preprocessed_data,  # List of strings for the "data" field
        vector_res  # List of embeddings, each is a list of floats
    ]

    # Insert data into Milvus
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
    print("Index created on 'embeddings' field.")

    collection.load()
    print("Collection is loaded and ready for search.")
