from services.query.status.states import QueryStates


class QueryStateController:
    _instance = None
    current_state = QueryStates.PENDING

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryStateController, cls).__new__(cls)
        return cls._instance

    def get_current_state(self):
        return self.current_state

    def set_current_state(self, state):
        if isinstance(state, QueryStates):
            self.current_state = state
        else:
            raise ValueError("Invalid state. Must be an instance of QueryStates.")
