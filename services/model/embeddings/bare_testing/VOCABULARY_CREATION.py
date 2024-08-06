from icecream import ic

from services.model.embeddings.corpus.json_encoder import JSONEncoder

if __name__ == '__main__':
    try:
        path = input("Enter the JSON Object file path: ")

        encoder = JSONEncoder(path)

        preprocessed = encoder.preprocess_for_encoding()

        vocab = encoder.create_vocab(preprocessed)

        ic(f"Preprocessed vocabulary: {preprocessed}\n")

        ic(f"vocabulary: {list(vocab.keys())}\n")

        ic(f"Length of vocabulary: {len(vocab)}\n")

    except Exception as e:
        print(e)