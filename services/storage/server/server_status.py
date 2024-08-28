import enum
from enum import Enum

'''
This class should be able to write and determine the status of write operation in Milvus instance server monitoring
the way that milvus works and behaves given an operation we should handle it with grace meaning not ending service or 
terminating the program and giving a good insight about what the issue was 
'''


class ServerStorageStatus(Enum):
    IS_AVAILABLE = True
    IS_BUSY = False
    HAS_ERROR = False

    def __init__(self):
        """
        Class for having record of what's the status of the server given that operations are being done
        """
        self.status = ServerStorageStatus.IS_AVAILABLE  # Default statis is available

    def change_state(self, state: Enum):
        """
        Chnage the state of the server given that operations are done or being made

        :param state:
        :return:
        """
        self.status = state

    def get_state(self) -> Enum:
        """
        Get the current state of the server
        :return status type: auto():
        """
        return self.status
