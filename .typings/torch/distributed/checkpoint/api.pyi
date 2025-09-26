import traceback as tb
from _typeshed import Incomplete

__all__ = ['CheckpointException']

WRAPPED_EXCEPTION = tuple[BaseException, tb.StackSummary]

class CheckpointException(BaseException):
    """Exception raised if failure was detected as part of a checkpoint load or save."""
    _failures: Incomplete
    def __init__(self, msg: str, failures: dict[int, WRAPPED_EXCEPTION]) -> None: ...
    @property
    def failures(self) -> dict[int, WRAPPED_EXCEPTION]:
        """Return a dictionary mapping node ranks to their associated exceptions in case of failure."""
    def __str__(self) -> str: ...
