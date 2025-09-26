import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import timedelta
from enum import Enum
from torch._C._distributed_c10d import ProcessGroup, Work as _Work, _SymmetricMemory
from torch.types import _device, _dtype, _int
from typing import overload

__all__ = ['empty', 'rendezvous', 'is_nvshmem_available']

class _ScaleMode(Enum):
    UNSCALED = 'unscaled'
    TENSOR_WISE = 'tensor-wise'
    ROW_WISE_SHARDED = 'row-wise-sharded'
    ROW_WISE_REPLICATED = 'row-wise-replicated'

class Work(_Work):
    event: Incomplete
    def __init__(self) -> None: ...
    def wait(self, timeout: timedelta = ...) -> bool: ...

@overload
def empty(*size: _int, dtype: _dtype | None = None, device: _device | None = None) -> torch.Tensor: ...
@overload
def empty(size: Sequence[_int], *, dtype: _dtype | None = None, device: _device | None = None) -> torch.Tensor: ...
def rendezvous(tensor: torch.Tensor, group: str | ProcessGroup) -> _SymmetricMemory:
    """
    rendezvous(tensor, group) -> _SymmetricMemory

    Establish a symmetric memory tensor among participating processes. This is
    a collective operation.

    Args:
        tensor (:class:`torch.Tensor`): the local tensor used to establish the symmetric memory tensor.
            It must be allocated via :func:`torch._distributed._symmetric_memory.empty()`. The shape,
            dtype, and device type must be identical across all participating processes.
        group (Union[str, :class:`torch.distributed.ProcessGroup`]): The group identifying the
            participating processes. This can be either a group name or a process group object.
    """
def is_nvshmem_available() -> bool:
    """
    is_nvshmem_available() -> bool

    Check if NVSHMEM is available in current build and on current system.
    """
