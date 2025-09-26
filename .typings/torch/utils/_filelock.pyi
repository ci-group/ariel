from _typeshed import Incomplete
from filelock import FileLock as base_FileLock
from torch.monitor import _WaitCounter as _WaitCounter
from types import TracebackType
from typing_extensions import Self

class FileLock(base_FileLock):
    """
    This behaves like a normal file lock.

    However, it adds waitcounters for acquiring and releasing the filelock
    as well as for the critical region within it.

    pytorch.filelock.enter - While we're acquiring the filelock.
    pytorch.filelock.region - While we're holding the filelock and doing work.
    pytorch.filelock.exit - While we're releasing the filelock.
    """
    region_counter: Incomplete
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None: ...
