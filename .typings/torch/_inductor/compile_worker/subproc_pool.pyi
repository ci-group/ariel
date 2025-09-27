from _typeshed import Incomplete
from concurrent.futures import Future, ProcessPoolExecutor
from enum import Enum
from torch._inductor import config as config
from torch._inductor.codecache import torch_key as torch_key
from torch._inductor.compile_worker.tracked_process_pool import TrackedProcessPoolExecutor as TrackedProcessPoolExecutor
from torch._inductor.compile_worker.utils import _async_compile_initializer as _async_compile_initializer
from torch._inductor.utils import get_ld_library_path as get_ld_library_path
from typing import Any, Callable, IO, TypeVar
from typing_extensions import Never, ParamSpec

log: Incomplete
_P = ParamSpec('_P')
_T = TypeVar('_T')

def _pack_msg(job_id: int, length: int) -> bytes: ...
def _unpack_msg(data: bytes) -> tuple[int, int]: ...

msg_bytes: Incomplete

def _send_msg(write_pipe: IO[bytes], job_id: int, job_data: bytes = b'') -> None: ...
def _recv_msg(read_pipe: IO[bytes]) -> tuple[int, bytes]: ...

class _SubprocExceptionInfo:
    """
    Carries exception info from subprocesses across the wire. traceback
    objects are not pickleable, so we store the trace as a string and
    use it for the message in the exception thrown in the main process.
    """
    details: Incomplete
    def __init__(self, details: str) -> None: ...

class SubprocException(Exception):
    """
    Thrown when a job in a subprocess raises an Exception.
    """
    def __init__(self, details: str) -> None: ...

class SubprocPickler:
    """
    Allows a caller to provide a custom pickler for passing data with the
    subprocess.
    """
    def dumps(self, obj: object) -> bytes: ...
    def loads(self, data: bytes) -> object: ...

class SubprocKind(Enum):
    FORK = 'fork'
    SPAWN = 'spawn'

class SubprocPool:
    """
    Mimic a concurrent.futures.ProcessPoolExecutor, but wrap it in
    a subprocess.Popen() to try to avoid issues with forking/spawning
    """
    pickler: Incomplete
    kind: Incomplete
    write_pipe: Incomplete
    read_pipe: Incomplete
    process: Incomplete
    write_lock: Incomplete
    read_thread: Incomplete
    futures_lock: Incomplete
    pending_futures: dict[int, Future[Any]]
    job_id_count: Incomplete
    running: bool
    def __init__(self, nprocs: int, pickler: SubprocPickler | None = None, kind: SubprocKind = ...) -> None: ...
    def submit(self, job_fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]: ...
    def _read_thread(self) -> None: ...
    def shutdown(self) -> None: ...

class SubprocMain:
    """Communicates with a SubprocPool in the parent process, called by __main__.py"""
    pickler: Incomplete
    kind: Incomplete
    read_pipe: Incomplete
    write_pipe: Incomplete
    write_lock: Incomplete
    nprocs: Incomplete
    pool: Incomplete
    running: bool
    def __init__(self, pickler: SubprocPickler, kind: SubprocKind, nprocs: int, read_pipe: IO[bytes], write_pipe: IO[bytes]) -> None: ...
    def _new_pool(self, nprocs: int, warm: bool) -> ProcessPoolExecutor: ...
    def main(self) -> None: ...
    def _shutdown(self) -> None: ...
    def submit(self, job_id: int, data: bytes) -> None: ...
    def _submit_inner(self, job_id: int, data: bytes) -> None: ...
    @staticmethod
    def do_job(pickler: SubprocPickler, data: bytes) -> bytes: ...
AnyPool = ProcessPoolExecutor | SubprocPool

def _warm_process_pool(pool: ProcessPoolExecutor, n: int) -> None: ...

class TestException(RuntimeError): ...

def raise_testexc() -> Never: ...
