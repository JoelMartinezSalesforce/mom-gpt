import logging

from manager.constants.manager_const import ManagerConstants
from manager.unit.crash_record import CrashRecord


class CrashRecorderWriter:
    def __init__(self, manager):
        self.logger = None
        self.manager = manager
        # Set up logging with the path from ManagerConstants
        self.setup_logging()

    def setup_logging(self):
        # Initialize the logger
        self.logger = logging.getLogger('CrashRecorderWriter')
        self.logger.setLevel(logging.INFO)  # Set level to info to capture all messages at this level and above

        # Define the path and file name for the log
        log_file_path = ManagerConstants.WRITE_PATH + 'crash_logs.txt'

        # Create file handler and set formatter
        handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Define the log message format
        handler.setFormatter(formatter)

        # Add handler to the logger
        self.logger.addHandler(handler)

    def write_crash_record(self, message, timestamp, location):
        # Write a new crash record
        new_record = CrashRecord(message, timestamp, location)
        self.manager.records.append(new_record)
        # Log the crash recording
        log_message = f"Crash recorded: {message} at {location} on {timestamp}"
        print(log_message)
        self.logger.info(log_message)  # Log the message to file
