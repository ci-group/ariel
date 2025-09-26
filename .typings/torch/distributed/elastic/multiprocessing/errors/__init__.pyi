from .error_handler import ErrorHandler as ErrorHandler
from .handlers import get_error_handler as get_error_handler
from _typeshed import Incomplete
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

__all__ = ['ProcessFailure', 'ChildFailedError', 'record', 'ErrorHandler', 'get_error_handler']

JSON = dict
_R = TypeVar('_R')
_P = ParamSpec('_P')

@dataclass
class ProcessFailure:
    """
    Represent the failed process result. When the worker process fails, it may record failure root cause into the file.

    Tries to read the failure timestamp from the provided ``error_file``,
    if the ``error_file`` does not exist, the timestamp is the current
    timestamp (seconds since epoch).

    The ``message`` field is a concise explanation of the failure. If
    the error file exists then the message is obtained from the error file.
    Otherwise one is generated based on the failure signature.

    .. note:: It is assumed that the ``error_file`` is written by
              ``torch.distributed.elastic.multiprocessing.errors.error_handler.ErrorHandler``.
              Otherwise the behavior is undefined.

    """
    local_rank: int
    pid: int
    exitcode: int
    error_file: str
    error_file_data: JSON = field(init=False)
    message: str = field(init=False)
    timestamp: int = field(init=False)
    def __post_init__(self) -> None: ...
    def _get_error_data(self, error_file_data: dict[str, Any]) -> tuple[str, int]: ...
    def _set_no_reply_file(self) -> None: ...
    def signal_name(self) -> str: ...
    def timestamp_isoformat(self):
        """Return timestamp in ISO format (YYYY-MM-DD_HH:MM:SS)."""
GlobalRank = int

class ChildFailedError(Exception):
    '''
    Special exception type that can be raised from a function annotated with the
    ``@record`` decorator to have the child process\' (root exception) propagate
    up the stack as-is (e.g. without being wrapped in the parent\'s traceback).

    Useful in cases where the parent is a simple nanny process
    and the child (worker) processes are actually doing meaningful compute.
    In this case, errors typically occur on the child process as the parent
    is not doing anything non-trivial, and child errors should be propagated
    to the scheduler for accurate root cause diagnostics.

    .. note:: The propagation relies on error files rather than exception handling to
              support both function and binary launches.

    Example:
    ::

     # process tree on a host (container)
     0: scheduler-init-process:
                |- 1: torchelastic_agent:
                         |- 2: trainer_0 (ok)
                         |- 3: trainer_1 (fail) -> error.json
                         |- ...
                         |- n+2: trainer_n (ok)
                |- n+3: other processes
                |- ...

    In the example above, trainer 1\'s failure (written into error.json) is
    the root cause and should be reported to the scheduler\'s init process.
    The torchelastic agent raises a ``ChildFailedError("trainer", {1: "trainer_1/error.json"})``
    upon detecting trainer 1\'s failure which would propagate the contents
    of trainer 1\'s error file to the scheduler\'s init process.
    '''
    name: Incomplete
    failures: Incomplete
    def __init__(self, name: str, failures: dict[GlobalRank, ProcessFailure]) -> None: ...
    def get_first_failure(self) -> tuple[GlobalRank, ProcessFailure]: ...
    def format_msg(self, boarder_delim: str = '=', section_delim: str = '-'): ...
    def _format_failure(self, idx: int, rank: int, failure: ProcessFailure) -> tuple[str, int]: ...

def record(fn: Callable[_P, _R], error_handler: ErrorHandler | None = None) -> Callable[_P, _R | None]:
    '''
    Syntactic sugar to record errors/exceptions that happened in the decorated
    function using the provided ``error_handler``.

    Using this decorator is equivalent to:

    ::

     error_handler = get_error_handler()
     error_handler.initialize()
     try:
         foobar()
     except ChildFailedError as e:
         _, failure = e.get_first_failure()
         error_handler.dump_error_file(failure.error_file, failure.exitcode)
         raise
     except Exception as e:
         error_handler.record_exception(e)
         raise

    .. important:: use this decorator once per process at the top level method,
                   typically this is the main method.

    Example

    ::

     @record
     def main():
         pass


     if __name__ == "__main__":
         main()

    '''
