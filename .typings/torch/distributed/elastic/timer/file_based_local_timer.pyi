import io
import threading
from _typeshed import Incomplete
from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

__all__ = ['FileTimerClient', 'FileTimerRequest', 'FileTimerServer']

_P = ParamSpec('_P')
_R = TypeVar('_R')

class FileTimerRequest(TimerRequest):
    '''
    Data object representing a countdown timer acquisition and release
    that is used between the ``FileTimerClient`` and ``FileTimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.
    ``signal`` is the signal to reap the worker process from the server
    process.
    '''
    __slots__: Incomplete
    version: int
    worker_pid: Incomplete
    scope_id: Incomplete
    expiration_time: Incomplete
    signal: Incomplete
    def __init__(self, worker_pid: int, scope_id: str, expiration_time: float, signal: int = 0) -> None: ...
    def __eq__(self, other) -> bool: ...
    def to_json(self) -> str: ...

class FileTimerClient(TimerClient):
    """
    Client side of ``FileTimerServer``. This client is meant to be used
    on the same host that the ``FileTimerServer`` is running on and uses
    pid to uniquely identify a worker.
    This client uses a named_pipe to send timer requests to the
    ``FileTimerServer``. This client is a producer while the
    ``FileTimerServer`` is a consumer. Multiple clients can work with
    the same ``FileTimerServer``.

    Args:

        file_path: str, the path of a FIFO special file. ``FileTimerServer``
                        must have created it by calling os.mkfifo().

        signal: signal, the signal to use to kill the process. Using a
                        negative or zero signal will not kill the process.
    """
    _file_path: Incomplete
    signal: Incomplete
    def __init__(self, file_path: str, signal=...) -> None: ...
    def _open_non_blocking(self) -> io.TextIOWrapper | None: ...
    def _send_request(self, request: FileTimerRequest) -> None: ...
    def acquire(self, scope_id: str, expiration_time: float) -> None: ...
    def release(self, scope_id: str) -> None: ...

class FileTimerServer:
    """
    Server that works with ``FileTimerClient``. Clients are expected to be
    running on the same host as the process that is running this server.
    Each host in the job is expected to start its own timer server locally
    and each server instance manages timers for local workers (running on
    processes on the same host).

    Args:

        file_path: str, the path of a FIFO special file to be created.

        max_interval: float, max interval in seconds for each watchdog loop.

        daemon: bool, running the watchdog thread in daemon mode or not.
                      A daemon thread will not block a process to stop.
        log_event: Callable[[Dict[str, str]], None], an optional callback for
                logging the events in JSON format.
    """
    _file_path: Incomplete
    _run_id: Incomplete
    _max_interval: Incomplete
    _daemon: Incomplete
    _timers: dict[tuple[int, str], FileTimerRequest]
    _stop_signaled: bool
    _watchdog_thread: threading.Thread | None
    _is_client_started: bool
    _request_count: int
    _run_once: bool
    _log_event: Incomplete
    _last_progress_time: Incomplete
    def __init__(self, file_path: str, run_id: str, max_interval: float = 10, daemon: bool = True, log_event: Callable[[str, FileTimerRequest | None], None] | None = None) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run_once(self) -> None: ...
    @staticmethod
    def is_process_running(pid: int):
        """
        function to check process is running or not
        """
    def _watchdog_loop(self) -> None: ...
    def _run_watchdog(self, fd: io.TextIOWrapper) -> None: ...
    def _get_scopes(self, timer_requests: list[FileTimerRequest]) -> list[str]: ...
    def _get_requests(self, fd: io.TextIOWrapper, max_interval: float) -> list[FileTimerRequest]: ...
    def register_timers(self, timer_requests: list[FileTimerRequest]) -> None: ...
    def clear_timers(self, worker_pids: set[int]) -> None: ...
    def get_expired_timers(self, deadline: float) -> dict[int, list[FileTimerRequest]]: ...
    def _reap_worker(self, worker_pid: int, signal: int) -> bool: ...
    def get_last_progress_time(self) -> int: ...
