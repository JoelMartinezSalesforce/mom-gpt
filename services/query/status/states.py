from enum import Enum


class QueryStates(Enum):
    """
    An enumeration representing the various states of a query process.

    Attributes:
        PENDING (str): The query is pending and has not yet started.
        CONNECTED (str): The query has successfully connected to the database.
        STARTED (str): The query process has started.
        RUNNING (str): The query is currently being executed.
        FAILED (str): The query has failed during execution.
        COMPLETED (str): The query has successfully completed.
    """
    PENDING = "PENDING"
    CONNECTED = "CONNECTED"
    STARTED = 'STARTED'
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
