from icecream import ic
from pymilvus import connections, Collection, SearchResult

from exceptions.query.query_exception import QueryException
from services.query.status.states import QueryStates
from services.query.status.trigger import QueryStateController


class QueryManager:
    def __init__(self, collection_name="health_data_cons_final"):
        self.collection_name = collection_name
        self.search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        self.state_controller = QueryStateController()
        self.connect()

    def connect(self):
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
        try:
            self.state_controller.set_current_state(QueryStates.RUNNING)
            collection = Collection(name=self.collection_name)
            self.state_controller.set_current_state(QueryStates.COMPLETED)
            return collection.search(vector, "embeddings", self.search_params, limit=5, output_fields=["id", "data"])
        except Exception as e:
            raise QueryException("Database connection failed", errors=str(e))
