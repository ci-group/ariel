from collections.abc import Generator
from contextlib import contextmanager

__all__ = ['is_available', 'range_push', 'range_pop', 'mark', 'range']

class _ITTStub:
    @staticmethod
    def _fail(*args, **kwargs) -> None: ...
    @staticmethod
    def is_available(): ...
    rangePush = _fail
    rangePop = _fail
    mark = _fail

def is_available():
    """
    Check if ITT feature is available or not
    """
def range_push(msg):
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Arguments:
        msg (str): ASCII message to associate with range
    """
def range_pop():
    """
    Pops a range off of a stack of nested range spans. Returns the
    zero-based depth of the range that is ended.
    """
def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Arguments:
        msg (str): ASCII message to associate with the event.
    """
@contextmanager
def range(msg, *args, **kwargs) -> Generator[None]:
    """
    Context manager / decorator that pushes an ITT range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
