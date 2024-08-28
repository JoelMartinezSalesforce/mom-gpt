from icecream import ic
from pymilvus import connections, Collection, SearchResult

from exceptions.query.query_exception import QueryException
from services.query.status.states import QueryStates
from services.query.status.trigger import QueryStateController


class QueryManager:
    """
    A class to manage querying operations on a Milvus database.

    Attributes:
        collection_name (str): The name of the collection to perform queries on.
        search_params (dict): The parameters used for searching the collection.
        state_controller (QueryStateController): A controller to manage query states.

    Methods:
        connect():
            Establishes a connection to the Milvus database.

        perform_search(vector) -> SearchResult:
            Executes a search on the collection using the provided vector and returns the results.
    """

    def __init__(self, collection_name="health_data_cons_final"):
        """
        Initializes the QueryManager with the specified collection name and sets up the connection to Milvus.

        Args:
            collection_name (str): The name of the Milvus collection to query. Defaults to "health_data_cons_final".
        """
        self.collection_name = collection_name
        self.search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        self.state_controller = QueryStateController()
        self.connect()

    def connect(self):
        """
        Establishes a connection to the Milvus database and updates the query state.

        The connection uses predefined credentials and connects to a Milvus instance running on localhost at port 19530.
        """
        connections.connect(
            alias="default",
            user='username',
            password='password',
            host='localhost',
            port='19530'
        )
        self.state_controller.set_current_state(QueryStates.CONNECTED)
        print("Connected to Milvus successfully.")

    def perform_search(self, vector) -> SearchResult:
        """
        Executes a search query on the Milvus collection using the provided vector.

        This method performs a vector search on the specified collection, retrieves up to 5 results, and includes the "id" and "data" fields in the output.

        Args:
            vector (list or numpy.ndarray): The vector to search for in the Milvus collection.

        Returns:
            SearchResult: The search results from the collection, including specified output fields.

        Raises:
            QueryException: If the database connection or search operation fails.
        """
        try:
            self.state_controller.set_current_state(QueryStates.RUNNING)
            collection = Collection(name=self.collection_name)
            self.state_controller.set_current_state(QueryStates.COMPLETED)
            return collection.search(vector, "embeddings", self.search_params, limit=5, output_fields=["id", "data"])
        except Exception as e:
            raise QueryException("Database connection failed", errors=str(e))
