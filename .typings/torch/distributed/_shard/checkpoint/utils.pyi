import io
import torch
import torch.distributed as dist
from .metadata import MetadataIndex, STATE_DICT_TYPE
from _typeshed import Incomplete
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

__all__ = ['find_tensor_shard', 'find_state_dict_object']

T = TypeVar('T')
R = TypeVar('R')

class _DistWrapper:
    """
    This is a wrapper around PG that provides a series of features around object collectives.

    It works without distributed initialized, where most collectives turns into nops.

    All variants that take functions are exception robust, meaning that if one or more
    ranks raise errors, all ranks will observe those.
    """
    group: Incomplete
    use_dist: Incomplete
    coordinator_rank: Incomplete
    global_coordinator_rank: Incomplete
    rank: Incomplete
    is_coordinator: Incomplete
    def __init__(self, group: dist.ProcessGroup | None, use_dist: bool, coordinator_rank: int) -> None: ...
    def get_rank(self) -> int: ...
    def get_world_size(self) -> int: ...
    def broadcast_object(self, object: T | None) -> T:
        """Implement functionality similar to c10d::broadcast_object_list but without distributed enabled."""
    def gather_object(self, object: T) -> list[T] | None:
        """Implement functionality similar to c10d::gather_object but without distributed enabled."""
    def all_gather_object(self, object: T) -> list[T]:
        """Implement functionality similar to c10d::all_gather_object but without distributed enabled."""
    def scatter_object(self, object_list: list[T] | None) -> T:
        """Implement functionality similar to c10d::scatter_object but without distributed enabled."""
    def reduce_scatter(self, step: str, map_fun: Callable[[], T], reduce_fun: Callable[[list[T]], list[R]]) -> R:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a scatter.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Scatter to each rank part of the result.
        """
    def all_reduce(self, step: str, map_fun: Callable[[], T], reduce_fun: Callable[[list[T]], R]) -> R:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a broadcast.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Broadcast the reduced value to all ranks.
        """
    def all_gather(self, step: str, map_fun: Callable[[], T]) -> list[T]:
        """
        Compute a value on each rank, then all_gather them.

        This method operates in the following way:
            Run ``map_cp`` on all ranks
            all_gather the values to all ranks
        """
    def broadcast(self, step: str, map_fun: Callable[[], T]) -> T:
        """
        Compute a value on rank 0 and broadcast it.

        This method operates in the following way:
            Run ``map_cp`` on rank 0
            broadcast the value
        """
    def barrier(self) -> None:
        """
        Add a synchronization point across all processes when using distributed.
        If torch.distributed is initialized, this function will invoke a barrier across the global process group.
        If torch.distributed is not initialized, this function is a no-op.
        """

def find_tensor_shard(tensor: torch.Tensor, index: MetadataIndex) -> torch.Tensor: ...
def find_state_dict_object(state_dict: STATE_DICT_TYPE, index: MetadataIndex) -> Any: ...

class _ReaderView(io.IOBase):
    offset: Incomplete
    len: Incomplete
    base_stream: Incomplete
    def __init__(self, base_stream: io.IOBase, offset: int, len: int) -> None: ...
    def seek(self, offset: int, whence: int = ..., /) -> int: ...
    def tell(self) -> int: ...
    def readable(self) -> bool: ...
    def seekable(self) -> bool: ...
    def readinto(self, b): ...
    def read(self, size: int = -1): ...
