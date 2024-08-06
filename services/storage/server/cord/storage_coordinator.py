from services.storage.server.write.storage_writer import StorageWriter


class StorageCoordinator:
    def __init__(self):
        self.writer = StorageWriter()
        print("init")