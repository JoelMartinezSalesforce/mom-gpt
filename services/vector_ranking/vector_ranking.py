import numpy as np
from typing import Dict, List, Optional

from numpy import ndarray
from pymilvus import utility, connections

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

    def generate_embeddings(self, prompt: str) -> Dict[Dict, ndarray]:
        """
        Generates embeddings for the user's prompt using each collection's embedding model.
        """
        prompt_embeddings = {}
        for collection, model in self.embedding_models.items():
            prompt_embedding = model.vectorize_texts([prompt])
            prompt_embeddings[collection] = prompt_embedding
        return prompt_embeddings

    def rank_collections(self, prompt: str) -> Optional[dict]:
        """
        Ranks collections based on the cosine similarity between the prompt embedding and each collection's
        model-generated embedding.
        """
        prompt_embeddings = self.generate_embeddings(prompt)
        rankings = []
        for collection, embedding in prompt_embeddings.items():
            # Assuming there's a standard way to compare embeddings directly, perhaps against a collection-specific
            # benchmark or centroid
            similarity = np.linalg.norm(embedding)  # Example placeholder for direct comparison metric
            rankings.append((collection, similarity))

        # Sort collections based on the comparison metric, highest first
        rankings.sort(key=lambda x: x[1], reverse=True)
        best_collection = rankings[0][0] if rankings else None
        return best_collection
