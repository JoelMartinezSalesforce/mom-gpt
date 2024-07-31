from enum import Enum


class QueryStates(Enum):
    PENDING = "PENDING"
    CONNECTED = "CONNECTED"
    STARTED = 'STARTED'
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
