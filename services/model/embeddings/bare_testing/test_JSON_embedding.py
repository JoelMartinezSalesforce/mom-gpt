import time
import csv
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.storage.gen.data_generator import MockDataGenerator
from tqdm import tqdm

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

    generator.create_new_dump(20)

    encoder = JSONEncoder(
        json_file_path="/Users/isaacpadilla/milvus-dir/mom-gpt/services/models/data/dump/data_dump.json"
    )

    preprocessed_data = encoder.preprocess_for_encoding()
    number_of_items = len(preprocessed_data)
    print("Preprocessed Data:", preprocessed_data)
    print(f"Number of preprocessed data: {number_of_items}")

    start_time = time.perf_counter()
    vector_results = []
    for text in tqdm(preprocessed_data, desc="Encoding"):
        vector_results.append(encoder.model_wrapper.encode(text))
    flattened_vector_results = vector_results  # Assuming encode returns a list of tensors
    print("Vector Results:", vector_results)
    print(f"Number of Embeddings Created: {len(vector_results)}")

    # Save embeddings to a CSV file immediately after creation
    csv_file_path = 'embeddings.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Embedding"])
        for idx, embedding in enumerate(vector_results):
            writer.writerow([idx, embedding.tolist()])  # Convert tensor to list if needed

    print(f"Embeddings saved to {csv_file_path}")

    # Set the embedding dimension based on the first result assuming all embeddings have the same dimension
    embedding_dim = vector_results[0].shape[1] if vector_results else 0

    # Define fields for the Milvus collection
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="data", dtype=DataType.JSON, max_length=4000),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]

    schema = CollectionSchema(fields, description="Network Data Embeddings")
    collection_name = "network_health_embeddings"
    if utility.has_collection(collection_name):
        health_embeddings = Collection(name=collection_name)
    else:
        health_embeddings = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")

    # Insert embeddings to Milvus
    print("Inserting Embeddings to Milvus...")
    entities = [
        [i for i in range(number_of_items)],
        [encoder.data],
        vector_results
    ]

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
    print(f"Time per Embedding Created: {total_elapsed_time / max(len(vector_results), 1):.4f} seconds")
    print(f"Total time taken for script execution: {total_elapsed_time:.4f} seconds")

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    print("Performing a single vector search...")
    res = health_embeddings.search([entities[-1][2]], "embeddings", search_params, limit=5, output_fields=["pk"])

    print(f"Results of the vector search: \n{res}")
