import abc
import signal
import torch.multiprocessing as mp
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntFlag
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from torch.distributed.elastic.multiprocessing.subprocess_handler import SubprocessHandler
from typing import Any, Callable

__all__ = ['DefaultLogsSpecs', 'SignalException', 'Std', 'to_map', 'RunProcsResult', 'PContext', 'get_std_cm', 'MultiprocessContext', 'SubprocessContext', 'LogsDest', 'LogsSpecs']

class SignalException(Exception):
    """
    Exception is raised inside the torchelastic agent process by the termination handler
    if the death signal got received by the process.
    """
    sigval: Incomplete
    def __init__(self, msg: str, sigval: signal.Signals) -> None: ...

class Std(IntFlag):
    NONE = 0
    OUT = 1
    ERR = 2
    ALL = OUT | ERR
    @classmethod
    def from_str(cls, vm: str) -> Std | dict[int, 'Std']:
        '''
        Example:
        ::

         from_str("0") -> Std.NONE
         from_str("1") -> Std.OUT
         from_str("0:3,1:0,2:1,3:2") -> {0: Std.ALL, 1: Std.NONE, 2: Std.OUT, 3: Std.ERR}

        Any other input raises an exception
        '''

def to_map(val_or_map: Std | dict[int, Std], local_world_size: int) -> dict[int, Std]:
    """
    Certain APIs take redirect settings either as a single value (e.g. apply to all
    local ranks) or as an explicit user-provided mapping. This method is a convenience
    method that converts a value or mapping into a mapping.

    Example:
    ::

     to_map(Std.OUT, local_world_size=2)  # returns: {0: Std.OUT, 1: Std.OUT}
     to_map({1: Std.OUT}, local_world_size=2)  # returns: {0: Std.NONE, 1: Std.OUT}
     to_map(
         {0: Std.OUT, 1: Std.OUT}, local_world_size=2
     )  # returns: {0: Std.OUT, 1: Std.OUT}
    """

@dataclass
class LogsDest:
    """
    For each log type, holds mapping of local rank ids to file paths.
    """
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)
    tee_stdouts: dict[int, str] = field(default_factory=dict)
    tee_stderrs: dict[int, str] = field(default_factory=dict)
    error_files: dict[int, str] = field(default_factory=dict)

class LogsSpecs(ABC, metaclass=abc.ABCMeta):
    """
    Defines logs processing and redirection for each worker process.

    Args:
        log_dir:
            Base directory where logs will be written.
        redirects:
            Streams to redirect to files. Pass a single ``Std``
            enum to redirect for all workers, or a mapping keyed
            by local_rank to selectively redirect.
        tee:
            Streams to duplicate to stdout/stderr.
            Pass a single ``Std`` enum to duplicate streams for all workers,
            or a mapping keyed by local_rank to selectively duplicate.
    """
    _root_log_dir: Incomplete
    _redirects: Incomplete
    _tee: Incomplete
    _local_ranks_filter: Incomplete
    def __init__(self, log_dir: str | None = None, redirects: Std | dict[int, Std] = ..., tee: Std | dict[int, Std] = ..., local_ranks_filter: set[int] | None = None) -> None: ...
    @abstractmethod
    def reify(self, envs: dict[int, dict[str, str]]) -> LogsDest:
        """
        Given the environment variables, builds destination of log files for each of the local ranks.

        Envs parameter contains env variables dict for each of the local ranks, where entries are defined in:
        :func:`~torchelastic.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent._start_workers`.
        """
    @property
    @abstractmethod
    def root_log_dir(self) -> str: ...

class DefaultLogsSpecs(LogsSpecs):
    """
    Default LogsSpecs implementation:

    - `log_dir` will be created if it doesn't exist
    - Generates nested folders for each attempt and rank.
    """
    _run_log_dir: Incomplete
    def __init__(self, log_dir: str | None = None, redirects: Std | dict[int, Std] = ..., tee: Std | dict[int, Std] = ..., local_ranks_filter: set[int] | None = None) -> None: ...
    @property
    def root_log_dir(self) -> str: ...
    def _make_log_dir(self, log_dir: str | None, rdzv_run_id: str): ...
    def reify(self, envs: dict[int, dict[str, str]]) -> LogsDest:
        """
        Uses following scheme to build log destination paths:

        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stdout.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stderr.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/error.json`
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

@dataclass
class RunProcsResult:
    """
    Results of a completed run of processes started with ``start_processes()``. Returned by ``PContext``.

    Note the following:

    1. All fields are mapped by local rank
    2. ``return_values`` - only populated for functions (not the binaries).
    3. ``stdouts`` - path to stdout.log (empty string if no redirect)
    4. ``stderrs`` - path to stderr.log (empty string if no redirect)

    """
    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)
    def is_failed(self) -> bool: ...

class PContext(abc.ABC, metaclass=abc.ABCMeta):
    """
    The base class that standardizes operations over a set of processes that are launched via different mechanisms.

    The name ``PContext`` is intentional to disambiguate with ``torch.multiprocessing.ProcessContext``.

    .. warning:: stdouts and stderrs should ALWAYS be a superset of
                 tee_stdouts and tee_stderrs (respectively) this is b/c
                 tee is implemented as a redirect + tail -f <stdout/stderr.log>
    """
    name: Incomplete
    entrypoint: Incomplete
    args: Incomplete
    envs: Incomplete
    stdouts: Incomplete
    stderrs: Incomplete
    error_files: Incomplete
    nprocs: Incomplete
    _stdout_tail: Incomplete
    _stderr_tail: Incomplete
    def __init__(self, name: str, entrypoint: Callable | str, args: dict[int, tuple], envs: dict[int, dict[str, str]], logs_specs: LogsSpecs, log_line_prefixes: dict[int, str] | None = None) -> None: ...
    def start(self) -> None:
        """Start processes using parameters defined in the constructor."""
    @abc.abstractmethod
    def _start(self) -> None:
        """Start processes using strategy defined in a particular context."""
    @abc.abstractmethod
    def _poll(self) -> RunProcsResult | None:
        '''
        Poll the run status of the processes running under this context.
        This method follows an "all-or-nothing" policy and returns
        a ``RunProcessResults`` object if either all processes complete
        successfully or any process fails. Returns ``None`` if
        all processes are still running.
        '''
    def wait(self, timeout: float = -1, period: float = 1) -> RunProcsResult | None:
        '''
        Wait for the specified ``timeout`` seconds, polling every ``period`` seconds
        for the processes to be done. Returns ``None`` if the processes are still running
        on timeout expiry. Negative timeout values are interpreted as "wait-forever".
        A timeout value of zero simply queries the status of the processes (e.g. equivalent
        to a poll).

        .. note::
            Multiprocessing library registers SIGTERM and SIGINT signal handlers that raise
            ``SignalException`` when the signals received. It is up to the consumer of the code
            to properly handle the exception. It is important not to swallow the exception otherwise
            the process would not terminate. Example of the typical workflow can be:

        .. code-block:: python
            pc = start_processes(...)
            try:
                pc.wait(1)
                .. do some other work
            except SignalException as e:
                pc.shutdown(e.sigval, timeout=30)

        If SIGTERM or SIGINT occurs, the code above will try to shutdown child processes by propagating
        received signal. If child processes will not terminate in the timeout time, the process will send
        the SIGKILL.
        '''
    @abc.abstractmethod
    def pids(self) -> dict[int, int]:
        """Return pids of processes mapped by their respective local_ranks."""
    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        """
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).
        """
    def close(self, death_sig: signal.Signals | None = None, timeout: int = 30) -> None:
        """
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).

        Args:
            death_sig: Death signal to terminate processes.
            timeout: Time to wait for processes to finish, if process is
                still alive after this time, it will be terminated via SIGKILL.
        """

def get_std_cm(std_rd: str, redirect_fn): ...

class MultiprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a function."""
    start_method: Incomplete
    _ret_vals: Incomplete
    _return_values: dict[int, Any]
    _pc: mp.ProcessContext | None
    _worker_finished_event: Incomplete
    def __init__(self, name: str, entrypoint: Callable, args: dict[int, tuple], envs: dict[int, dict[str, str]], start_method: str, logs_specs: LogsSpecs, log_line_prefixes: dict[int, str] | None = None) -> None: ...
    def _start(self) -> None: ...
    def _is_done(self) -> bool: ...
    def _poll(self) -> RunProcsResult | None: ...
    def pids(self) -> dict[int, int]: ...
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None: ...

class SubprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a binary."""
    _running_local_ranks: set[int]
    _failures: dict[int, ProcessFailure]
    subprocess_handlers: dict[int, SubprocessHandler]
    def __init__(self, name: str, entrypoint: str, args: dict[int, tuple], envs: dict[int, dict[str, str]], logs_specs: LogsSpecs, log_line_prefixes: dict[int, str] | None = None) -> None: ...
    def _start(self) -> None: ...
    def _poll(self) -> RunProcsResult | None: ...
    def pids(self) -> dict[int, int]: ...
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None: ...
