from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from sklearn.feature_extraction.text import TfidfVectorizer

from services.model.embeddings.corpus.json_encoder import JSONEncoder

if __name__ == '__main__':
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    print("Collections in the system:", utility.list_collections())

    utility.drop_collection('health_data_cons_final')

    print("Collections in the system:", utility.list_collections())

    encoder = JSONEncoder(
        json_file_path=
        "/Users/isaacpadilla/milvus-dir/mom-gpt/services/model/embeddings/bare_testing/dump/network_health_cons.json"
    )

    COLLECTION_NAME = "health_data_cons_final"
    preprocessed_data = encoder.preprocess_for_encoding()

    print(preprocessed_data[0])

    vocab = encoder.create_vocab(preprocessed_data, 0.5)

    print(vocab)

    vectorizer = TfidfVectorizer(vocabulary=vocab, max_features=len(vocab))
    vector_res = vectorizer.fit_transform(preprocessed_data).toarray()

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    embedding_dim = vector_res.shape[1]
    print(f"Embedding dimensions {embedding_dim}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]

    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    if utility.has_collection(COLLECTION_NAME):
        health_embeddings = Collection(name=COLLECTION_NAME)
    else:
        health_embeddings = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"Collection '{COLLECTION_NAME}' created.")

    print("Inserting Embeddings to Milvus...")

    entities = [
        preprocessed_data,  # text
        vector_res.tolist()  # vector
    ]

    insert_result = health_embeddings.insert(entities)

    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    health_embeddings.create_index("embeddings", index)
    health_embeddings.load()

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 5},
    }
