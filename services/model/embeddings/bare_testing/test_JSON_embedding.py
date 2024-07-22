import time
import csv
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.storage.gen.data_generator import MockDataGenerator

if __name__ == '__main__':
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    print("Collections in the system:", utility.list_collections())

    generator = MockDataGenerator({
        "period": "month",
        "instance": "2024-05",
        "site": "FislXFzr6Z",
        "metric": "internet-avail-cc",
        "end": "2021-06-01 00:00:00",
        "start": "2021-05-01 00:00:00",
        "updated": "2024-06-01 00:00:00",
        "percentage-overall": 69.60791174172184,
        "percentage-EUROPE": 19.24033096521821,
        "percentage-NORTH_AMERICA": 74.04259099467755,
        "percentage-ASIA": 689
    })

    generator.create_new_dump(6)

    encoder = JSONEncoder(
        json_file_path="/Users/isaacpadilla/milvus-dir/mom-gpt/services/models/data/dump/data_dump.json"
    )

    preprocessed_data = encoder.preprocess_for_encoding()
    number_of_items = len(preprocessed_data)
    print("Preprocessed Data:", preprocessed_data)
    print(f"Number of preprocessed data: {number_of_items}")

    start_time = time.perf_counter()

    # Encoding the list of texts directly
    vector_res = encoder.model_wrapper.encode(preprocessed_data)

    print("Vector Results:", [elem.tolist for elem in vector_res])
    print(f"Number of Embeddings Created: {len(vector_res)}")

    # Save embeddings to a CSV file immediately after creation
    csv_file_path = 'embeddings.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Text", "Embedding"])
        for idx, (text, embedding) in enumerate(zip(preprocessed_data, vector_res)):
            writer.writerow([idx, text, embedding.tolist()])  # Convert tensor to list if needed

    print(f"Embeddings saved to {csv_file_path}")

    # Set the embedding dimension based on the first result assuming all embeddings have the same dimension
    COLLECTION_NAME = "health_data"
    # print(vector_res[0].tolist())
    embedding_dim = encoder.model_wrapper.encoding_dimensions
    print(embedding_dim)

    # Check that the collection does not yet exist
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Define fields for the Milvus collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="data", dtype=DataType.JSON),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]

    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    if utility.has_collection(COLLECTION_NAME):
        health_embeddings = Collection(name=COLLECTION_NAME)
    else:
        health_embeddings = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"Collection '{COLLECTION_NAME}' created.")
    # Insert embeddings to Milvus
    print("Inserting Embeddings to Milvus...")
    entities = [
        preprocessed_data,
        [elem.tolist() for elem in vector_res]
    ]

    # inserting fields into milvus
    insert_result = health_embeddings.insert(entities)

    # Create an index for faster search
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    health_embeddings.create_index("embeddings", index)
    health_embeddings.load()

    # Calculate execution time
    total_elapsed_time = time.perf_counter() - start_time
    print(f"Time per Embedding Created: {total_elapsed_time / max(len(vector_res), 1):.4f} seconds")
    print(f"Total time taken for script execution: {total_elapsed_time:.4f} seconds")

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    print("Performing a single vector search...")
    res = health_embeddings.search(vector_res[:1], "embeddings", search_params, limit=5, output_fields=["pk", "data"])

    print(f"Results of the vector search: \n{res}")
