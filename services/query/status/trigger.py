from services.query.status.states import QueryStates


class QueryStateController:
    """
    A singleton class to control and track the state of a query process.

    Attributes:
        _instance (QueryStateController): The singleton instance of the class.
        current_state (QueryStates): The current state of the query process, initialized to PENDING.

    Methods:
        __new__(cls):
            Ensures only one instance of the class is created (singleton pattern).

        get_current_state():
            Retrieves the current state of the query process.

        set_current_state(state):
            Updates the current state of the query process, ensuring the state is a valid instance of QueryStates.
    """
    _instance = None
    current_state = QueryStates.PENDING

    def __new__(cls):
        """
        Creates a new instance of the class if one does not exist; otherwise, returns the existing instance.

        Returns:
            QueryStateController: The singleton instance of the QueryStateController class.
        """
        if cls._instance is None:
            cls._instance = super(QueryStateController, cls).__new__(cls)
        return cls._instance

    def get_current_state(self):
        """
        Retrieves the current state of the query process.

        Returns:
            QueryStates: The current state of the query.
        """
        return self.current_state

    def set_current_state(self, state):
        """
        Sets the current state of the query process.

        Args:
            state (QueryStates): The new state to be set. Must be an instance of the QueryStates enum.

        Raises:
            ValueError: If the provided state is not an instance of QueryStates.
        """
        if isinstance(state, QueryStates):
            self.current_state = state
        else:
            raise ValueError("Invalid state. Must be an instance of QueryStates.")
