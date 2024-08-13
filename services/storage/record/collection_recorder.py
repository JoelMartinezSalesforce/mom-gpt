import json
import threading
from pymilvus import connections, utility


class CollectionRecord:
    def __init__(self, milvus_host='localhost', milvus_port='19530', interval=60):
        """
        Initializes the connection to Milvus server and sets up a periodic check for collection names.

        Args:
            milvus_host (str): Host address of the Milvus server.
            milvus_port (str): Port number for the Milvus server.
            interval (int): Interval in seconds at which to check for updates in collection names.
        """
        # Establish connection to Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        self.interval = interval
        self.filename = 'collections.json'
        self.collection_names = []
        self.update_collections()

    def update_collections(self):
        """
        Updates the list of collection names from Milvus and saves to a JSON file.
        """
        try:
            current_collections = utility.list_collections()
            if set(current_collections) != set(self.collection_names):
                self.collection_names = current_collections
                self.save_collections_to_json()
        except Exception as e:
            print(f"Failed to update collections: {str(e)}")
        finally:
            # Set the timer to call itself again after the specified interval
            threading.Timer(self.interval, self.update_collections).start()

    def save_collections_to_json(self):
        """
        Saves the current list of collection names to a JSON file.
        """
        try:
            with open(self.filename, 'w') as file:
                json.dump(self.collection_names, file)
            print(f"Collection names successfully updated and saved to {self.filename}.")
        except Exception as e:
            print(f"Failed to save collection names to JSON file: {str(e)}")

    def get_collection_names(self):
        """
        Returns the current list of Milvus collection names.

        Returns:
            list: A list of current Milvus collection names.
        """
        return self.collection_names
