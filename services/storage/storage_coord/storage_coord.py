class StorageCoord(object):
    """
    This class should be able to coordinate the way that milvus uses persistent storage for log and index files
    Storage Coord should know what Milvus and our service is doing so that its able to efficiently not create
    or create vector indexes based on the type and is able to coordinate
    """
    _instances = {}

    def __init__(self):
        print("Storage Coord")

    def __call__(self, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if self not in self._instances:
            instance = super().__call__(*args, **kwargs)
            self._instances[self] = instance
        return self._instances[self]

