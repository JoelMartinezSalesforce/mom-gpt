import os

from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from tqdm import tqdm

from manager.admin.crash_manager import CrashRecorderManager
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
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    print("Collections in the system:", utility.list_collections())
    num_samples = 500

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
    )  
    # encoder = JSONEncoder()

    # generator.create_new_dump(num_samples)

    manager = CrashRecorderManager()
    manager.manage_json_embedding("/Users/isaacpadilla/milvus-dir/mom-gpt/services/models/data/dump/data_dump.json")
