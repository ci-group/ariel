import abc
import contextlib
import torch
import torch.distributed as dist
import weakref
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle
from typing import Any, Protocol

__all__ = ['context_parallel', 'set_rotate_method']

class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True

class _RotateMethod(Enum):
    ALL_TO_ALL = ...
    ALL_GATHER = ...

class _DispatchMode(Enum):
    MONKEY_PATCH = ...
    TORCH_FUNCTION = ...
    TORCH_DISPATCH = ...

@dataclass
class _ContextParallelOptions:
    convert_to_f32: bool = ...
    enable_load_balance = ...
    rotate_method: _RotateMethod = ...

class _SDPAMerger:
    """A class to help to merge the local SDPA result."""
    _seq_dim: Incomplete
    _out: torch.Tensor | None
    _lse: torch.Tensor | None
    _convert_to_f32: Incomplete
    _out_dtype: Incomplete
    _lse_dtype: Incomplete
    def __init__(self, convert_to_f32: bool, seq_dim: int) -> None: ...
    def _merge_one(self, block_out: torch.Tensor, block_lse: torch.Tensor, partial: bool) -> None: ...
    def step(self, out: torch.Tensor, lse: torch.Tensor, partial: bool) -> None: ...
    def results(self) -> tuple[torch.Tensor, torch.Tensor]: ...

class _AttentionOp(Protocol):
    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, ...]: ...

class _RingRotater(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int): ...
    @abstractmethod
    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None: ...
    @abstractmethod
    def next_buffer(self) -> torch.Tensor: ...

class _AllToAllRotater(_RingRotater):
    """Use all_to_all to send the kv to the next rank"""
    _pg: Incomplete
    _seq_dim: Incomplete
    _buffer: torch.Tensor | None
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None: ...
    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None: ...
    def next_buffer(self) -> torch.Tensor: ...

class _AllGatherRotater(_RingRotater):
    """
    Allgather the kv and return the only the requried kv.
    Only one communication will be done.
    """
    _pg: Incomplete
    _seq_dim: Incomplete
    _aggregated_buffer: torch.Tensor | None
    _idx: int
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None: ...
    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None: ...
    def next_buffer(self) -> torch.Tensor: ...

class _AttentionContextParallel(ParallelStyle):
    """
    Applies context parallel optimizations to the attention layer.

    This will work for nn.MultiHeadedAttention and custom attention layers that
    call F.scaled_dotproduct_attention with a simliar signature.

    This expects the `forward` method consumes either:

    * a single tensor for self attention
    * one argument for each of: query, key, value

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """
    _CONTEXT_MANAGERS: weakref.WeakKeyDictionary[nn.Module, Any]
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    @classmethod
    def _input_fn(cls, module: nn.Module, inputs: tuple[torch.Tensor | int | float, ...], device_mesh: DeviceMesh) -> tuple[torch.Tensor | int | float, ...]: ...
    @classmethod
    def _output_fn(cls, module: nn.Module, outputs: torch.Tensor | tuple[torch.Tensor | int | float, ...], device_mesh: DeviceMesh) -> torch.Tensor | int | float | tuple[torch.Tensor | int | float, ...]: ...

class _LoadBalancer(ABC, metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def shard(cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int) -> torch.Tensor: ...
    @classmethod
    @abstractmethod
    def unshard(cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int) -> torch.Tensor: ...

class _SequentialSharder(_LoadBalancer):
    """
    This load balancer chunks the buffer into cp_world_size and rank0 gets
    0th shard, rank1 gets 1st shard, ...
    So this doesn't have any load balancing effect when using the causal masking.
    """
    @classmethod
    def shard(cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int) -> torch.Tensor: ...
    @classmethod
    def unshard(cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int) -> torch.Tensor: ...

class _RoundRobinLoadBalancer(_LoadBalancer):
    """
    This load balancer chunk the buffer into cp_world_size * ROUND_ROBIN_CYCLE
    shards, and uses a round robin approach to achieve load balancing.
    Since ROUND_ROBIN_CYCLE being 2 will achieve perfect load balancing for
    causal masking, we assume ROUND_ROBIN_CYCLE is always 2 to simplify the
    implementation.
    """
    ROUND_ROBIN_CYCLE: int
    @classmethod
    def shard(cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int) -> torch.Tensor: ...
    @classmethod
    def unshard(cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int) -> torch.Tensor: ...

@contextlib.contextmanager
def context_parallel(mesh: DeviceMesh, *, buffers: list[torch.Tensor] | None = None, buffer_seq_dims: list[int] | None = None, no_restore_buffers: set[torch.Tensor] | None = None) -> Generator[None, None, None]:
    """

    ``context_parallel`` is an experimental API to enable context
    parallelism (CP). This API performs two actions: 1) patch the SDPA
    (``torch.nn.functional.scaled_dot_product_attention``) with the CP-enabled
    one, 2) shard ``buffers`` along the sequence dimension and each rank will
    preserve the corresponding shard according ``mesh``.

    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (Optional[List[torch.Tensor]]): buffers that the usage depend
            on the sequence dimension. Examples are input batch, labels and
            positional embedding buffers. These buffers must be sharded along
            the sequence dimension to ensure the accuracy. The sharding will
            happen in-place, the buffer's shape will change within the context.
            The buffers will be restored after the context finishes.
            ``no_restore_buffers`` can be used to specify which buffers don't
            need to be restored. Note that ``buffers`` should not contain any
            nn.Parameter.
        buffer_seq_dims (Optional[List[int]]): the sequence dimensions of ``buffers``.
        no_restore_buffers (Optional[Set[torch.Tensor]]): buffers in these set
            won't be restored after the context exits. This set must be a subset
            of ``buffers``. If the buffers won't be used after the context exits,
            these buffers can be put in this list to avoid extra restore time.

    .. warning::
        `torch.distributed.tensor.experimental.context_parallel` is a
        prototype feature in PyTorch. The API is subject to change.
    """
def set_rotate_method(rotate_method: str) -> None:
    '''
    Context Parallel SDPA requires the rotation of kv shards. Users can call this
    API to specify which rotation method to use. "alltoall" shuffles the kv shards
    using all-to-all collective. While "allgather" gathers the kv shards using
    all-gather collective after the first sub-SDPA computation. If this API has not
    been called, the default rotate method is "allgather".

    Args:
        rotate_method (str): the rotate method to use. Currently only supports
        "allgather" and "alltoall". If a different string other than these two
        is passed in, the function will raise an error.

    Returns:
        None
    '''
