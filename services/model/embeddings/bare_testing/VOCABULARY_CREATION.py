from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.embeddings.corpus.vocab import VocabularyCreator

if __name__ == '__main__':
    encoder = JSONEncoder(
        '/Users/isaacpadilla/milvus-dir/mom-gpt/services/model/embeddings/bare_testing/dump/network_health_cons.json'
    )

    creator = VocabularyCreator(ngram_range=(1, 2))
    preprocessed_texts = encoder.preprocess_for_encoding()

    print(preprocessed_texts[0])

    vocabulary_1 = creator.create_vocab(preprocessed_texts)
    # vocabulary = encoder.create_vocab(preprocessed_texts)
    print(list(vocabulary_1.keys()))
