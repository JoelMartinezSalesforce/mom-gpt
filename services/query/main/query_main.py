from icecream import ic
from pymilvus import SearchResult
from sklearn.feature_extraction.text import TfidfVectorizer

from exceptions.query.query_exception import QueryException
from services.model.constants.embedding_const import EmbeddingConstants
from services.query.manager.manager import QueryManager


class Query:
    def __init__(self, collection_name: str):
        # Initialize the QueryManager
        self.query_manager = QueryManager(collection_name)
        print("Query system initialized. Ready to receive queries.")

    def take_user_input(self):
        # Take input from the user
        return input("Enter your search prompt: ")

    def perform_query(self) -> SearchResult:
        try:
            # Get user input
            prompt = self.take_user_input()

            # Process the input to generate a vector
            # Assuming the QueryManager's 'perform_search' expects a pre-processed vector
            vectorizer = TfidfVectorizer(max_features=329, vocabulary=EmbeddingConstants.VOCABULARY)
            vector = vectorizer.fit_transform([prompt]).toarray()

            # Execute the search
            ic("Query completed successfully.")
            return self.query_manager.perform_search(vector)

        except QueryException as e:
            ic(f"An error occurred: {e}")

        except Exception as e:
            ic(f"An unexpected error occurred: {e}")

    def ingest(self) -> SearchResult:
        # Main method to start the query process
        print("Starting query process...")
        return self.perform_query()
