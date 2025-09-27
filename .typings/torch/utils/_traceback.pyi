import contextlib
from _typeshed import Incomplete
from collections.abc import Generator

@contextlib.contextmanager
def report_compile_source_on_error() -> Generator[None]: ...
def shorten_filename(fn, *, base=None):
    """Shorten a source filepath, with the assumption that torch/ subdirectories don't need to be shown to user."""
def format_frame(frame, *, base=None, line: bool = False):
    """
    Format a FrameSummary in a short way, without printing full absolute path or code.

    The idea is the result fits on a single line.
    """
def format_traceback_short(tb):
    """Format a TracebackType in a short way, printing only the inner-most frame."""

class CapturedTraceback:
    __slots__: Incomplete
    tb: Incomplete
    skip: Incomplete
    def __init__(self, tb, skip: int = 0) -> None: ...
    def cleanup(self) -> None: ...
    def summary(self): ...
    def __getstate__(self): ...
    @staticmethod
    def extract(*, script: bool = False, cpp: bool = False, skip: int = 0):
        """
        Like traceback.extract_stack(), but faster (approximately 20x faster); it
        is fast enough that you can unconditionally log stacks this way as part of
        normal execution.  It returns a torch._C._profiler.CapturedTraceback
        object that must be formatted specially with format_captured_tb.

        By default, this only reports Python backtraces (like extract_stack).  You
        can set the script/cpp kwargs to also turn on TorchScript/C++ trace
        reporting.
        """
    def format(self):
        """
        Formats a single torch._C._profiler.CapturedTraceback into a list of
        strings equivalent to the output of traceback.format_list.  Note that if
        pass it CapturedTraceback with C++ traces,  it is better not to use this
        function and use the batch formatting API format_captured_tbs to amortize
        the cost of symbolization
        """
    @staticmethod
    def format_all(tbs):
        """
        Bulk version of CapturedTraceback.format.  Returns a list of list of strings.
        """

def _extract_symbolized_tb(tb, skip):
    """
    Given a symbolized traceback from symbolize_tracebacks, return a StackSummary object of
    pre-processed stack trace entries.
    """
