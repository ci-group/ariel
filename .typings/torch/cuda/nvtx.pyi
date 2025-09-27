from collections.abc import Generator
from contextlib import contextmanager

__all__ = ['range_push', 'range_pop', 'range_start', 'range_end', 'mark', 'range']

class _NVTXStub:
    @staticmethod
    def _fail(*args, **kwargs) -> None: ...
    rangePushA = _fail
    rangePop = _fail
    markA = _fail

def range_push(msg):
    """
    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
    """
def range_pop():
    """Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended."""
def range_start(msg) -> int:
    """
    Mark the start of a range with string message. It returns an unique handle
    for this range to pass to the corresponding call to rangeEnd().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns: A range handle (uint64_t) that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
    """
def range_end(range_id) -> None:
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): an unique handle for the start range.
    """
def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
@contextmanager
def range(msg, *args, **kwargs) -> Generator[None]:
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
