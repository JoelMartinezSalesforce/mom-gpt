import os

from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from tqdm import tqdm

from models.data.gen.data_generator import MockDataGenerator
from models.embeddings.corpus.json_encoder import JSONEncoder


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
    json_file_path = "../dump/data_dump.json"
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    # Generate JSON data and save it to a file
    generator.create_new_dump(num_samples)

    # Encode data using threading
    def encode_data(path):
        return encoder.encode_json_data(path)

    # Single thread for encoding since it processes the entire file
    embeddings = encode_data(json_file_path)

    return embeddings


if __name__ == '__main__':
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    print("Collections in the system:", utility.list_collections())
    num_samples = 1000

    generator = MockDataGenerator(
        {
            "period": "month",
            "instance": "2024-05",
            "site": "ntN06ZyAIg",
            "metric": "internet-avail-cc",
            "end": "2021-06-01 00:00:00",
            "start": "2021-05-01 00:00:00",
            "updated": "2024-06-01 00:00:00",
            "percentage-overall": 61.00315908896376,
            "percentage-EUROPE": 38.970381437105715,
            "percentage-NORTH_AMERICA": 39.50339959335143,
            "percentage-ASIA": 306
        }
    )  # Initialize generator
    encoder = JSONEncoder()  # Initialize encoder

    print("Generating and encoding data...")
    embeddings = generate_and_encode_data(generator, encoder, num_samples)

    collection_name = "network_health_embeddings"
    dim = len(embeddings[0]) if embeddings.size > 0 else 0  # Determine dimension from first embedding
    collection = create_collection(collection_name, dim)

    print("Inserting embeddings into Milvus...")
    insert_embeddings(collection, embeddings)

    print("All operations completed successfully.")
    print("Collections in the system:", utility.list_collections())
