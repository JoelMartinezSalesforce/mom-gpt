class StorageException(Exception):
    """Custom exception class to handle query-related errors."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
        self.message = message

    def __str__(self):
        error_details = f": {self.errors}" if self.errors else ""
        return f"{self.message}{error_details}"
