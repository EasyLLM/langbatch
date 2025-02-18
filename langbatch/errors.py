class SetupError(Exception):
    pass

class BatchError(Exception):
    """Raised when batch operations fail"""
    pass

class BatchInitializationError(BatchError):
    """Raised when batch initialization fails"""
    pass

class BatchStateError(BatchError):
    """Raised when batch operations are attempted in invalid states"""
    pass

class BatchValidationError(BatchError):
    """Raised when batch validation fails"""
    pass

class BatchStartError(BatchError):
    """Raised when batch start fails"""
    pass

class BatchStorageError(BatchError):
    """Raised when batch storage operations fail"""
    pass

class BatchResultsError(BatchError):
    """Raised when batch fetch operations fail"""
    pass