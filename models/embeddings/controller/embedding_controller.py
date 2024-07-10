"""
This class should handle the functionality of the controller so that the Milvus framework can only call creation
of embedding when needed from a given data input

this will control when to call the embedding model efficiently check or data or existing data given the structure of
where this data should be
"""


class EmbeddingController:
    def __init__(self):
        print("Initiallizing controller ")
