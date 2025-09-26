import torch.distributed as dist
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar('T')

@dataclass
class SyncPayload(Generic[T]):
    stage_name: str | None
    success: bool
    payload: T
    exception: Exception | None = ...

def broadcast(data_or_fn: T | Callable[[], T], *, success: bool = True, stage_name: str | None = None, rank: int = 0, pg: dist.ProcessGroup | None = None) -> T:
    """
    Broadcasts the data payload from rank 0 to all other ranks.
    Or if a function is passed, execute it in rank 0 and broadcast result to all other ranks.

    Can be used to broadcast a failure signal to stop all ranks.

    If the function raises an exception, all ranks will raise.

    Args:
        data_or_fn: the data to broadcast or function to execute and broadcast result.
        success: False to stop all ranks.
        stage_name: the name of the logical stage for synchronization and debugging
        rank: rank to broadcast data or execute function and broadcast results.
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        the value after synchronization

    Example usage:
    >> id = broadcast(data_or_fn=allocate_id, rank=0, pg=ext_pg.my_pg)
    """
def all_gather(data_or_fn: T | Callable[[], T], stage_name: str | None = None, pg: dist.ProcessGroup | None = None) -> list[T]:
    """
    A simple all_gather primitive with basic synchronization guard logic,
    by checking payload from all ranks has the same stage name.

    Args:
        data_or_fn: the data to be all gathered across ranks or function to be executed
        stage_name: the sync stage name for out-of-sync protection
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        a list of synced data from all ranks

    Example usage:
    >> all_ids = all_gather(data_or_fn=allocate_id, pg=ext_pg.my_pg)
    """
def all_gather_object_enforce_type(pg: dist.ProcessGroup, object_list: list[Any], obj: Any, type_checker: Callable[[Any, Any], bool] = ...) -> None:
    """
    Similar to plain all_gather_object but with additional type checking
    AFTER gather is done to ensure basic consistency.
    If check does not pass, all ranks will fail with exception.

    This is generally to prevent conditional logic leading to
    unexpected messages being received. This is considered fatal code error,
    but due to logic stacks this might happen implicitly in practice.

    The default check does not check sub type (considered different)
    or covariance (considered same) but users can pass in custom checker
    if more complicated check is needed.
    """
