import numpy as np
from typing import Dict, List, Tuple, Any

from icecream import ic
from numpy import ndarray
from pymilvus import utility, connections
from sklearn.metrics.pairwise import cosine_similarity

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.local.vectorizer import VectorizerEmbedding


class VectorRanking:
    def __init__(self):
        """
        Initializes the VectorRanking class by loading vocabularies for each collection and
        initializing embedding models for those vocabularies.
        """
        self.vocabularies = {item['name']: item['vocab'] for item in self._getVocabs()}
        self.embedding_models = {
            collection: VectorizerEmbedding(vocab) for collection, vocab in self.vocabularies.items()
        }

    def get_vocabs(self):
        return self.vocabularies

    def _getVocabs(self) -> List[Dict[str, Dict]]:
        """
        Retrieves the vocabularies for each collection from the Milvus database.

        Returns:
            List[Dict[str, Dict]]: A list of dictionaries, each containing the collection name and its corresponding vocabularies.
        """
        # Connect to the Milvus database
        connections.connect(
            alias="default",
            user='username',
            password='password',
            host='localhost',
            port='19530'
        )

        # List all collections
        collections = utility.list_collections()
        encoder = JSONEncoder()

        # Create a list of dictionaries, each containing a collection name and its vocabularies
        collections_vocabs_list = []
        for collection in collections:
            vocab_name = collection + "_vocab"
            vocab = encoder.get_vocab(vocab_name)
            collections_vocabs_list.append({
                "name": collection,
                "vocab": vocab
            })

        return collections_vocabs_list

    def generate_embeddings(self, prompt: str) -> Dict[Dict, ndarray[Any, Any]]:
        """
        Generates embeddings for the user's prompt using each collection's embedding model.
        """
        prompt_embeddings = {}
        for collection, model in self.embedding_models.items():
            # Assuming vectorize_texts returns an embedding array
            prompt_embedding = model.vectorize_texts([prompt])[0]  # Take the first (and only) embedding
            prompt_embeddings[collection] = prompt_embedding
        return prompt_embeddings

    def pad_or_trim(self, vector: ndarray, target_length: int) -> ndarray:
        """
        Pads or trims the vector to match the target length.

        :param vector: The original vector.
        :param target_length: The desired length of the vector.
        :returns: The padded or trimmed vector.
        """
        current_length = len(vector)
        if current_length < target_length:
            # Pad with zeros if the current length is less than the target length
            return np.pad(vector, (0, target_length - current_length), 'constant')
        elif current_length > target_length:
            # Trim if the current length exceeds the target length
            return vector[:target_length]
        else:
            return vector

    def rank_collections(self, prompt: str, n_shots: int = 5) -> List[Tuple[str, float]]:
        """
        Ranks collections based on the cosine similarity between the prompt embedding and each collection's
        model-generated embedding.

        :param prompt: The user-provided text prompt.
        :param n_shots: The number of times to run the similarity scoring and aggregate results.
        :returns: A list of tuples containing the collection name and its aggregated cosine similarity score.
        """

        # Generate prompt embeddings for each collection
        prompt_embeddings = self.generate_embeddings(prompt)

        # Calculate the "internal ratio" for each embedding (proportion of non-zero elements)
        internal_ratios = {k: np.count_nonzero(v) / len(v) for k, v in prompt_embeddings.items()}

        # Sort embeddings by internal ratio in descending order
        sorted_embeddings = sorted(prompt_embeddings.items(), key=lambda item: internal_ratios[item[0]], reverse=True)

        # Determine the reference vector (highest internal ratio)
        reference_vector = sorted_embeddings[0][1]
        reference_dim = len(reference_vector)

        # Normalize and adjust embeddings to the reference vector length
        normalized_embeddings = {}
        for k, v in sorted_embeddings:
            # Pad or trim the embedding to match the reference dimension
            adjusted_v = self.pad_or_trim(v, reference_dim)
            normalized_v = adjusted_v / np.linalg.norm(adjusted_v)  # Normalize
            normalized_embeddings[k] = normalized_v

        rankings = []

        # Compute similarity and aggregate scores
        for collection, embedding in normalized_embeddings.items():
            aggregated_score = 0

            # Run the similarity n_shots times and aggregate the results
            for _ in range(n_shots):
                similarity = \
                cosine_similarity([normalized_embeddings[next(iter(normalized_embeddings))]], [embedding])[0][0]
                aggregated_score += similarity

            # Boost the score if the collection name or its parts are in the prompt
            collection_words = collection.split()  # Split collection name into words
            for word in collection_words:
                if any(word.lower() in prompt_word.lower() for prompt_word in prompt.split()):
                    aggregated_score += 0.1  # Arbitrary boost value
                    break  # Apply boost only once per collection

            rankings.append((collection, aggregated_score))

        # Sort collections based on the aggregated score, highest first
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

