from icecream import ic

from services.model.embeddings.corpus.json_encoder import JSONEncoder

if __name__ == '__main__':
    try:
        path = input("Enter the JSON Object file path: ")

        encoder = JSONEncoder()

        ic(f"Retrieving a vocab from a collection mame: 'network_health_cons_vocab'")

        res = encoder.get_vocab('network_health_cons_vocab')

        ic(res)

    except Exception as e:
        print(e)
