class CrashRecorderReader:
    def __init__(self, manager):
        self.manager = manager

    def read_last_record(self):
        # Read the most recent crash record
        if self.manager.records:
            return self.manager.records[-1]
        else:
            return None