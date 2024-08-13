import logging
from typing import List, Dict

from nltk.lm import Vocabulary
from pymilvus import SearchResult

from exceptions.query.query_exception import QueryException
from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.local.vectorizer import VectorizerEmbedding
from services.query.manager.manager import QueryManager

# Configure logging at the module level
logging.basicConfig(
    filename=f'{__name__}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class Query:
    _logger = logging.getLogger(__name__)  # This retrieves a logger configured at the module level

    def __init__(self, collection_name: str, vocabulary: Dict[str, any]):
        self.query_manager = QueryManager(collection_name)
        self._logger.info("Query system initialized. Ready to receive queries.")
        self.vocabulary = vocabulary

    def perform_query(self, prompt) -> List[Dict[str, any]]:
        try:
            # Process the input to generate a vector
            vectorizer = VectorizerEmbedding(vocabulary=self.vocabulary)
            vector = vectorizer.vectorize_texts([prompt])

            # Execute the search
            result = self.query_manager.perform_search(vector)
            self._logger.info("Query executed successfully.")
            return [elem.to_dict() for elem in list(result[0])]

        except QueryException as e:
            self._logger.error(f"An error occurred due to query processing: {e}", exc_info=True)
            raise

        except Exception as e:
            self._logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            raise

    def ingest(self, prompt: str) -> List[Dict[str, any]]:
        # Main method to start the query process
        self._logger.info("Starting query process...")
        return self.perform_query(prompt)
