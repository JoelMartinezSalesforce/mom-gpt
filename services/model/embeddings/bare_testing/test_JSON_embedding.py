import os
import time

from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from tqdm import tqdm

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.storage.gen.data_generator import MockDataGenerator


def create_collection(collection_name, dim, alias="default"):
    connections.connect(alias=alias)
    fields = [
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Network Data Embeddings")
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    return collection


def insert_embeddings(collection, embeddings):
    batch_size = 1000
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Inserting embeddings"):
        batch = embeddings[i:i + batch_size]
        entities = [{"embeddings": emb.tolist()} for emb in batch]
        collection.insert(entities)
    collection.load()


def generate_and_encode_data(generator, encoder, num_samples):
    json_file_path = "dump/data_dump.json"
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    generator.create_new_dump(num_samples)

    def encode_data(path):
        return encoder.encode_json_data(path)

    embeddings = encode_data(json_file_path)

    return embeddings


if __name__ == '__main__':
    start_time = time.perf_counter()

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

    encoder = JSONEncoder(
        json_file_path="/Users/isaacpadilla/milvus-dir/mom-gpt/services/models/data/dump/data_dump.json"
    )

    preprocessed_data = encoder.preprocess_for_encoding()
    number_of_items = len(preprocessed_data)
    print("Preprocessed Data:", preprocessed_data)
    print(f"Number of preprocessed data: {number_of_items}")

    vector_results = [encoder.model_wrapper.encode(text) for text in preprocessed_data]
    flattened_vector_results = vector_results  # Assuming encode returns a list of tensors
    print("Vector Results:", vector_results)
    print(f"Number of Embeddings Created: {len(vector_results)}")

    # Set the embedding dimension based on the first result assuming all embeddings have the same dimension
    embedding_dim = vector_results[0].shape[1] if vector_results else 0

    # Define fields for the Milvus collection
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
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
    res = health_embeddings.search(entities[-1], "embeddings", search_params, limit=5, output_fields=["pk"])

    print(f"Results of the vector search: \n{res}")
