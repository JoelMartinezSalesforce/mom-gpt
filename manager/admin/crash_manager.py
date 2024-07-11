import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from manager.reader.crash_reader import CrashRecorderReader
from manager.writter.crash_writer import CrashRecorderWriter
from services.model.constants.embedding_const import EmbeddingConstants
from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.embeddings.embedding_model import EmbeddingModelWrapper
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


def _get_current_timestamp():
    return datetime.now().isoformat()


def insert_embeddings(collection, embeddings):
    batch_size = 1000
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Inserting embeddings", unit="batch"):
        batch = embeddings[i:i + batch_size]
        entities = [{"embeddings": emb.tolist()} for emb in batch]
        collection.insert(entities)
    collection.load()


class CrashRecorderManager:
    def __init__(self):
        self.logger = None
        self.records = []
        self.writer = CrashRecorderWriter(self)
        self.reader = CrashRecorderReader(self)
        self.encoder = JSONEncoder(EmbeddingConstants.SALESFORCE_2_R)
        self.embedding_model = EmbeddingModelWrapper.instance(EmbeddingConstants.SALESFORCE_2_R, 256)

    def connect_to_milvus(self):
        try:
            connections.connect(
                alias="default",
                user='username',
                password='password',
                host='localhost',
                port='19530'
            )
            print("Connected to Milvus")
        except Exception as e:
            self.flag_crash(f"Milvus connection error: {str(e)}", "@CrashConnectionMilvus")

    def flag_crash(self, message, location):
        timestamp = _get_current_timestamp()
        self.writer.write_crash_record(message, timestamp, location)

    def review_last_crash(self):
        last_record = self.reader.read_last_record()
        print(f"Reviewing crash: {last_record.message} at {last_record.location} on {last_record.timestamp}")
        return last_record.location

    def manage_json_embedding(self, json_data_path=None):
        if json_data_path is None:
            self.logger.info("No JSON data path provided. Aborting embedding process.")
            return

        self.connect_to_milvus()
        data = self.load_and_preprocess_data(json_data_path)
        if not data:
            self.logger.info("No data to process after preprocessing.")
            return

        embeddings = self.create_embeddings(data)
        self.store_embeddings_in_milvus(embeddings)

    def load_and_preprocess_data(self, json_data_path):
        try:
            data = self.encoder.load_json_file(json_data_path)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file: {json_data_path}. Error: {e}")
            return []

        preprocessed_data = []
        for item in data:
            item_string = json.dumps(item)
            processed_text = self.encoder.preprocess_text(item_string)
            if processed_text.strip():
                preprocessed_data.append(processed_text)

        return preprocessed_data

    def create_embeddings(self, preprocessed_data):
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(executor.map(self.embedding_model.encode, preprocessed_data), total=len(preprocessed_data),
                     desc="Creating embeddings"))
        return results

    def store_embeddings_in_milvus(self, embeddings):
        if not embeddings:
            self.logger.info("No embeddings to store in Milvus.")
            return

        collection_name = "health_embedding"
        vector_dim = 256

        fields = [FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)]
        schema = CollectionSchema(fields, description="Network Data Embeddings")

        try:
            if not utility.has_collection(collection_name):
                collection = Collection(name=collection_name, schema=schema)
                self.logger.info(f"Collection '{collection_name}' created.")
            else:
                collection = Collection(name=collection_name)
                self.logger.info(f"Collection '{collection_name}' already exists.")

            insert_embeddings(collection, embeddings)
        except Exception as e:
            self.logger.error(f"Failed to create or access the Milvus collection: {e}")
