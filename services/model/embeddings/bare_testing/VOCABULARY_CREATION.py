from icecream import ic

from services.model.embeddings.corpus.json_encoder import JSONEncoder

if __name__ == '__main__':
    try:
        path = '/Users/isaacpadilla/milvus-dir/mom-gpt/services/model/embeddings/bare_testing/dump/network_health_cons.json'

        encoder = JSONEncoder(path)

        pre_processed = encoder.preprocess_for_encoding()

        ic(pre_processed[:2])

        vocab = encoder.create_vocab(pre_processed)

        ic(f"Retrieving a vocab from a collection mame: 'network_health_cons_vocab'")

        res = encoder.get_vocab('network_health_cons')

        ic(res)

    except Exception as e:
        print(e)
