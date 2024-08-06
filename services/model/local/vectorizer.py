from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from services.model.constants.embedding_const import EmbeddingConstants
from services.model.embeddings.corpus.json_encoder import JSONEncoder


class VectorizerEmbedding:
    def __init__(self, encoder: JSONEncoder):
        """
        Initialize the VectorizerEmbedding with a predefined vocabulary.
        """

        self.preprocessed = encoder.preprocess_for_encoding()
        self.vocab = list(encoder.create_vocab(preprocessed_texts=self.preprocessed).keys())
        self.vectorizer = TfidfVectorizer(vocabulary=self.vocab, max_features=len(self.vocab))

    def vectorize_texts(self, texts):
        """
        Converts a list of preprocessed text data into TF-IDF vectors using the predefined vocabulary.

        Args:
        texts (list[str]): Preprocessed text data to be vectorized.

        Returns:
        ndarray: Array of TF-IDF vectors.
        """
        # Transform the texts to TF-IDF features
        vector_res = self.vectorizer.fit_transform(texts).toarray()
        return vector_res

    def update_vocabulary(self, new_vocab):
        """
        Updates the vectorizer with a new vocabulary and adjusts the maximum number of features.

        Args:
        new_vocab (list[str]): A new vocabulary for vectorization.
        """
        self.vocab = new_vocab
        self.vectorizer = TfidfVectorizer(vocabulary=self.vocab, max_features=len(self.vocab))
