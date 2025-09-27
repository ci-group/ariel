import atexit
import dataclasses
from _typeshed import Incomplete
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

_P = ParamSpec('_P')
_R = TypeVar('_R')
log: Incomplete

@dataclass
class _QueueStats:
    pending: dict[int, float] = dataclasses.field(default_factory=dict)
    timing: list[float] = dataclasses.field(default_factory=list)
    enqueue_count: int = ...
    dequeue_count: int = ...
    max_queue_depth: int = ...
    pool_count: int = ...

_queue_stats: Incomplete
_queue_stats_lock: Incomplete

class TrackedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, max_workers: int | None = None, mp_context: BaseContext | None = None, initializer: Callable[[], object] | None = None) -> None: ...
    def _record_dequeue(self, f: Future[Any]) -> None: ...
    def _record_enqueue(self, f: Future[Any]) -> None: ...
    def submit(self, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_R]: ...

@atexit.register
def _queue_stats_report() -> None: ...
