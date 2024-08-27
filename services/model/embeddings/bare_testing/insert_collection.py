from icecream import ic
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from sklearn.feature_extraction.text import TfidfVectorizer

from services.model.embeddings.bare_testing.test_summarization import NLPSummarization
from services.model.embeddings.corpus.json_encoder import JSONEncoder


def batch_insert(collection, data, batch_size=100):
    for i in range(0, len(data), batch_size):
        ic(f"Batch: {i + 1}")
        batch = [d[i: i + batch_size] for d in data]
        collection.insert(batch)


if __name__ == '__main__':
    connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
    )

    print("Collections in the system:", utility.list_collections())

    utility.drop_collection('network_health_cons')

    print("Collections in the system:", utility.list_collections())

    '''
    Extract to const
    '''

    encoder = JSONEncoder(
        json_file_path=
        "/Users/joel.martinez/Documents/mom-gpt-master/services/model/embeddings/bare_testing/dump/network_health_cons.json"
    )

    COLLECTION_NAME = "network_health_cons"
    preprocessed_data = encoder.preprocess_for_encoding()

    summarizer = NLPSummarization()

    print(preprocessed_data[0])

    vocab = list(encoder.create_vocab(preprocessed_data).keys())

    # Summarization and Cropping
    MAX_LENGTH = 65534
    summarized = [
        " ".join(summarizer.summarize_and_extract_based_on_keywords(elem, vocab))[:MAX_LENGTH] if len(
            elem) > MAX_LENGTH else elem[:MAX_LENGTH]
        for elem in preprocessed_data
    ]

    ic(max(len(elem) for elem in summarized))

    print(vocab)

    vectorizer = TfidfVectorizer(vocabulary=vocab, max_features=len(vocab))
    vector_res = vectorizer.fit_transform(summarized).toarray()
    embedding_dim = vector_res.shape[1]
    print(f"Embedding dimensions {embedding_dim}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=MAX_LENGTH),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]

    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    print(f"Collection '{COLLECTION_NAME}' created.")

    print("Inserting Embeddings to Milvus...")

    # Define batch size
    BATCH_SIZE = 1000

    # Inserting data in batches
    entities = [
        summarized,
        vector_res.tolist()
    ]

    batch_insert(collection, entities, BATCH_SIZE)

    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    collection.create_index("embeddings", index)
    collection.load()
