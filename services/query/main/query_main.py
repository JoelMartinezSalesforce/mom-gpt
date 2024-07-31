import logging

from icecream import ic
from pymilvus import SearchResult

from exceptions.query.query_exception import QueryException
from services.model.constants.embedding_const import EmbeddingConstants
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

    def __init__(self, collection_name: str):
        self.query_manager = QueryManager(collection_name)
        self._logger.info("Query system initialized. Ready to receive queries.")

    def perform_query(self, prompt) -> SearchResult:
        try:
            # Process the input to generate a vector
            vectorizer = VectorizerEmbedding(EmbeddingConstants.VOCABULARY)
            vector = vectorizer.vectorize_texts([prompt])

            # Execute the search
            result = self.query_manager.perform_search(vector)
            self._logger.info("Query executed successfully.")
            return result

        except QueryException as e:
            self._logger.error(f"An error occurred due to query processing: {e}", exc_info=True)
            raise

        except Exception as e:
            self._logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            raise

    def ingest(self, prompt: str) -> SearchResult:
        # Main method to start the query process
        self._logger.info("Starting query process...")
        return self.perform_query(prompt)
