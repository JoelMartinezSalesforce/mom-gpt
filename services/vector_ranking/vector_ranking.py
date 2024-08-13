import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from numpy import ndarray
from pymilvus import utility, connections
from sklearn.metrics.pairwise import cosine_similarity

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.local.vectorizer import VectorizerEmbedding


class VectorRanking:
    def __init__(self):
        """
        Initializes the class with vocabularies for each collection and a dictionary of embedding models.
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

    def rank_collections(self, prompt: str) -> List[Tuple[Dict, Any]]:
        """
        Ranks collections based on the cosine similarity between the prompt embedding and each collection's
        model-generated embedding.

        :returns List[Tuple[str, float]]: A list of tuples containing the collection name and its cosine similarity.
        """
        prompt_embeddings = self.generate_embeddings(prompt)
        max_dim = max(len(v) for v in prompt_embeddings.values())

        # Normalize and pad embeddings to ensure all are the same dimension
        normalized_embeddings = {k: np.pad(v / np.linalg.norm(v), (0, max_dim - len(v)), 'constant')
                                 for k, v in prompt_embeddings.items()}

        rankings = []
        prompt_vector = normalized_embeddings[next(iter(normalized_embeddings))]  # Use the first collection's vector as reference
        for collection, embedding in normalized_embeddings.items():
            # Calculate cosine similarity
            similarity = cosine_similarity([prompt_vector], [embedding])[0][0]
            rankings.append((collection, similarity))

        # Sort collections based on similarity, highest first
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

