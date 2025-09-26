from enum import Enum

class ExportErrorType(Enum):
    INVALID_INPUT_TYPE = 1
    INVALID_OUTPUT_TYPE = 2
    VIOLATION_OF_SPEC = 3
    NOT_SUPPORTED = 4
    MISSING_PROPERTY = 5
    UNINITIALIZED = 6

def internal_assert(pred: bool, assert_msg: str) -> None:
    """
    This is exir's custom assert method. It internally just throws InternalError.
    Note that the sole purpose is to throw our own error while maintaining similar syntax
    as python assert.
    """

class InternalError(Exception):
    """
    Raised when an internal invariance is violated in EXIR stack.
    Should hint users to report a bug to dev and expose the original
    error message.
    """
    def __init__(self, message: str) -> None: ...

class ExportError(Exception):
    """
    This type of exception is raised for errors that are directly caused by the user
    code. In general, user errors happen during model authoring, tracing, using our public
    facing APIs, and writing graph passes.
    """
    def __init__(self, error_code: ExportErrorType, message: str) -> None: ...
