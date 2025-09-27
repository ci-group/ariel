import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
import weakref
from .metadata import ShardedTensorMetadata as ShardedTensorMetadata, TensorProperties as TensorProperties
from .reshard import reshard_local_shard as reshard_local_shard, reshuffle_local_shard as reshuffle_local_shard
from .shard import Shard as Shard
from .utils import _flatten_tensor_size as _flatten_tensor_size, _parse_and_validate_remote_device as _parse_and_validate_remote_device, _validate_output_tensor_for_gather as _validate_output_tensor_for_gather, build_global_metadata as build_global_metadata, build_metadata_from_local_shards as build_metadata_from_local_shards
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass
from torch._utils import _get_device_module as _get_device_module
from torch.distributed import distributed_c10d as distributed_c10d, rpc as rpc
from torch.distributed._shard._utils import DEPRECATE_MSG as DEPRECATE_MSG
from torch.distributed._shard.metadata import ShardMetadata as ShardMetadata
from torch.distributed._shard.sharding_spec._internals import check_tensor as check_tensor, validate_non_overlapping_shards_metadata as validate_non_overlapping_shards_metadata
from torch.distributed._shard.sharding_spec.api import _dispatch_custom_op as _dispatch_custom_op, _has_custom_op as _has_custom_op
from torch.distributed.remote_device import _remote_device as _remote_device
from typing import Callable

_sharded_tensor_lock: Incomplete
_sharded_tensor_current_id: int
_sharded_tensor_map: dict[int, weakref.ReferenceType[ShardedTensor]]
_SHARDED_OPS: dict[Callable, Callable]
_CUSTOM_SHARDED_OPS: dict[Callable, Callable]

def _register_remote_shards(sharded_tensor_id: int, rrefs: list[rpc.RRef[Shard]], rpc_rank: int): ...

class ShardedTensorBase(torch.Tensor):
    _sharding_spec: shard_spec.ShardingSpec
    _metadata: ShardedTensorMetadata
    _local_shards: list[Shard]
    def __new__(cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs): ...
    def metadata(self) -> ShardedTensorMetadata:
        """
        Returns a :class:`ShardedTensorMetadata` object corresponding to the
        metadata for the entire tensor.
        """
    def local_shards(self) -> list[Shard]:
        """
        Returns a list of :class:`Shard' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
    @classmethod
    def _init_from_local_shards_and_global_metadata(cls, local_shards: list[Shard], sharded_tensor_metadata: ShardedTensorMetadata, sharding_spec=None) -> ShardedTensorBase:
        """
        Initialize a ShardedTensorBase with local shards and a global
        ShardedTensorMetadata built on each rank.
        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None) -> None: ...

class ShardedTensor(ShardedTensorBase):
    """
    ShardedTensor is an torch.Tensor subclass to represent Tensors that are sharded
    across multiple devices and multiple processes.

    ShardedTensor is initialized in an SPMD like fashion where each rank
    initializes the ShardedTensor. The ShardedTensor object on each rank
    then only stores the local shard for the Tensor and provides global
    metadata for all the shards.

    ShardedTensor doesn't provide any Tensor like operations but is a wrapper
    providing the Tensor representing the local shard and the global metadata.
    Using these, users can build their custom distributed._sharded computations
    on top of this primitive. The local shards are all initialized using the
    create_op specified by tensor_init_params.create_op, e.g., torch.ones, or
    torch.empty

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    .. note:: ShardedTensor uses collectives to do various operations, i.e. it
        uses all_gather to do cross rank validations. For NCCL-based process
        groups, internal tensor representations of objects must be moved to the
        GPU device before communication takes place. In this case, the device
        used is given by ``torch.cuda.current_device()`` and it is the user's
        responsibility to ensure that this is set so that each rank has an
        individual GPU, via ``torch.cuda.set_device()``

    """
    def __new__(cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs): ...
    def __init__(self, sharding_spec: shard_spec.ShardingSpec, *size, dtype=None, layout=..., requires_grad: bool = False, pin_memory: bool = False, memory_format=..., process_group=None, init_rrefs: bool = False) -> None: ...
    _init_rrefs: Incomplete
    _sharded_tensor_id: Incomplete
    _process_group: Incomplete
    _remote_shards: dict[int, list[rpc.RRef[Shard]]]
    def _prepare_init(self, process_group=None, init_rrefs: bool = False) -> None: ...
    def _post_init(self) -> None: ...
    def __del__(self) -> None: ...
    def _init_rpc(self) -> None: ...
    def _get_preferred_device(self) -> torch.device:
        """
        Return the preferred device to be used when creating tensors for collectives.
        This method takes into account the associated process group
        """
    def gather(self, dst: int = 0, out: torch.Tensor | None = None, enforce_dtype: bool = False, dtype: torch.dtype | None = None) -> None:
        """
        Creates a full :class:`Tensor` on rank ``dst`` by gathering all shards of the
        sharded tensor.

        The API needs to be called on all ranks in SPMD fashion. All ranks should have
        the same ``dst``. ``out`` should be a tensor of the same size as the overall
        size of the sharded tensor on ``dst`` and ``None`` on all other ranks.

        Args:
            dst(int): The rank where full tensor is constructed.
                Default: 0
            out (:class `torch.Tensor`, optional): The output full tensor.
                Must to be provided ONLY on ``dst`` rank.
                Default: ``None``
            enforce_dtype (bool): Deprecated, please use dtype instead.  Force the
                gathered tensors to be the same type as input and output.
            dtype (torch.dtype): Force the gathered tensors to be this dtype.
                Default: ``None``
        """
    def cpu(self, memory_format=..., process_group=None) -> ShardedTensor:
        """
        Returns a copy of this object in CPU memory.

        If this ShardedTensor is already on CPU memory, then no copy is
        performed and original object is returned.

        .. note:: When moving a ShardedTensor from GPU to CPU, the ShardedTensor might
            need to be managed by a different type of ProcessGroup(i.e. ProcessGroupGloo),
            it is the user's responsibility to explicitly pass in a new process_group that
            is compatible with CPU.
        """
    def cuda(self, device=None, non_blocking: bool = False, memory_format=..., process_group=None) -> ShardedTensor:
        """
        Returns a copy of this object in CUDA memory, if the original ShardedTensor
        is on CPU, we will move the local shard to the current GPU device of each
        process in a SPMD fashion.
        If this ShardedTensor is already on CUDA memory and local shards on each rank are
        already on current device, we still returns a new ShardedTensor object with new
        metadata, but no underlying data movements are performed.
        .. note:: When moving a ShardedTensor from CPU to GPU, the ShardedTensor might
            need to be managed by a different type of ProcessGroup(i.e. ProcessGroupNCCL),
            it is the user's responsibility to explicitly pass in a new process_group that
            is compatible with GPU.
        """
    def to(self, *args, **kwargs) -> ShardedTensor: ...
    @classmethod
    def _normalize_pg(cls, process_group: dist.ProcessGroup | None) -> dist.ProcessGroup: ...
    @classmethod
    def _init_from_local_shards(cls, local_shards: list[Shard], *global_size, process_group=None, init_rrefs: bool = False): ...
    @classmethod
    def _init_from_local_tensor(cls, local_tensor: torch.Tensor, sharding_spec: shard_spec.ShardingSpec, *global_size: Sequence[int], process_group: dist.ProcessGroup | None = None, init_rrefs: bool = False) -> ShardedTensor:
        '''
        Initialize a ShardedTensor given only one local tensor, global sharded tensor
        size and sharding spec on each rank.

        Args:
            local_tensor (Tensor): Single tensor of local shard stored in each rank.
            sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
                The specification describing how to shard the Tensor.
            global_size (Sequence[int]): Size of the sharded tensor.
            process_group (ProcessGroup, optional): The process group to aggregate on.
                Default: None
            init_rrefs (bool, optional): Whether or not to initialize
                :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
                Need to initialize the RPC Framework if specified as ``True``.
                Default: ``False``.

        Returns:
            A :class:`ShardedTensor` sharded based on the given sharding_spec with local
                tensor stored in the current rank.

        Examples:
            >>> # xdoctest: +SKIP
            >>> # All tensors below are of torch.int64 type.
            >>> # We have 2 process groups, 2 ranks.
            >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
            >>> local_tensor = torch.unsqueeze(torch.cat([tensor, tensor + 2]))
            >>> local_tensor
            tensor([[1, 2, 3, 4]]) # Rank 0
            tensor([[3, 4, 5, 6]]) # Rank 1
            >>> sharding_dim = 0
            >>> sharding_spec = ChunkShardingSpec(
                    dim=sharding_dim,
                    placements=[
                        "rank:0/cuda:0",
                        "rank:1/cuda:1",
                    ],
                )
            >>> st = ShardedTensor._init_from_local_tensor(
            ...     local_tensor, sharding_spec, [2, 4]
            ... )
            >>> st
            ShardedTensor(
                ShardedTensorMetadata(
                    shards_metadata=[
                        ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1, 4], placement=rank:0/cuda:0),
                        ShardMetadata(shard_offsets=[1, 0], shard_sizes=[1, 4], placement=rank:1/cuda:1),
                    ],
                    size=torch.Size([2, 4])
            )
            >>> st.local_tensor()
            tensor([1, 2, 3, 4]) # Rank 0
            tensor([3, 4, 5, 6]) # Rank 1

        Warning: This API is experimental and subject to change. It lacks of a fully across
                 rank validations, and we only validate the local shard on the current rank.
                 We fully rely on the user to ensure local tensor is sharded based on the
                 sharding spec.
        '''
    @classmethod
    def _init_from_local_shards_and_global_metadata(cls, local_shards: list[Shard], sharded_tensor_metadata: ShardedTensorMetadata, process_group=None, init_rrefs: bool = False, sharding_spec=None) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
    def sharding_spec(self) -> shard_spec.ShardingSpec:
        """
        Returns the ShardingSpec for the tensor.
        """
    _local_shards: Incomplete
    _sharding_spec: Incomplete
    def reshard(self, resharding_spec: shard_spec.ShardingSpec) -> ShardedTensor:
        '''
        Reshard a sharded tensor given the ``resharding_spec``. For now, we only support
        single local shard.

        If ``resharding_spec`` is same as the original one, this becomes a no-op.
        If only ``resharding_spec`` shares the same sharding dim with the original one,
        we swap local shards directly.
        For more generic cases, we merge different shards across different ranks and split
        the local shards based on the ``resharding_spec`` via `all_to_all` collective API.

        Args:
            resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
                specification describing how the tensor is sharded.

        Returns:
            A :class:`ShardedTensor` object whose local shards are resharded.

        Examples:
            >>> # xdoctest: +SKIP
            >>> # We have 2 process groups, 2 ranks.
            >>> tensor = torch.arange(4, dtype=torch.int64) + 1 + 2 * rank
            >>> tensor = torch.stack([tensor, tensor])
            >>> tensor
            tensor([[1, 2, 3, 4], [1, 2, 3, 4]]) # Rank 0
            tensor([[3, 4, 5, 6], [3, 4, 5, 6]]) # Rank 1
            tensor([[5, 6, 7, 8], [5, 6, 7, 8]]) # Rank 2
            tensor([[7, 8, 9, 10], [7, 8, 9, 10]]) # Rank 3
            >>> sharding_dim = 0
            >>> spec = ChunkShardingSpec(
                    dim=sharding_dim,
                    placements=[
                        "rank:0/cuda:0",
                        "rank:1/cuda:1",
                        "rank:2/cuda:2",
                        "rank:3/cuda:3",
                    ],
                )
            >>> current_offsets = [0] * 2
            >>> current_offsets[0] = rank * 2
            >>> shard_metadata = ShardMetadata(
                    shard_offsets=copy.deepcopy(current_offsets),
                    shard_sizes=tensor.size(),
                    placement=spec.placements[rank],
                )
            >>> local_shards = [
                    Shard(
                        tensor=tensor,
                        metadata=shard_metadata,
                    )
                ]
            >>> st = ShardedTensor._init_from_local_shards(local_shards, tensor.size())
            >>> sharding_dim = 1
            >>> resharding_spec = ChunkShardingSpec(
                    dim=sharding_dim,
                    placements=[
                        "rank:0/cuda:0",
                        "rank:1/cuda:1",
                        "rank:2/cuda:2",
                        "rank:3/cuda:3",
                    ],
                )
            >>> st.reshard(resharding_spec)
            >>> tensor = st.local_shards()[0].tensor
            >>> tensor
            tensor([[1], [1], [3], [3], [5], [5], [7], [7]]) # Rank 0
            tensor([[2], [2], [4], [4], [6], [6], [8], [8]]) # Rank 1
            tensor([[3], [3], [5], [5], [7], [7], [9], [9]]) # Rank 2
            tensor([[4], [4], [6], [6], [8], [8], [10], [10]]) # Rank 3
        '''
    def local_tensor(self) -> torch.Tensor:
        """
        Return local tensor for a sharded_tensor. For now we only support single local shard.

        Returns:
            A :class:`torch.Tensor` of the local shard.
        """
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None): ...
    def is_pinned(self) -> bool:
        """
        Returns True if the sharded tensor (each local shard) resides in pinned memory.
        """
    def _register_remote_shards(self, remote_shards: list[rpc.RRef[Shard]], rpc_rank: int): ...
    def remote_shards(self) -> dict[int, list[rpc.RRef[Shard]]]:
        """
        Returns a Dict[int, RRef] with keys being the RPC rank and values
        being RRefs to shards on that rank. Need to initialize the
        RPC framework for this functionality.

        Raises an exception if ShardedTensor was created with ``init_rrefs=False``
        """
    def __hash__(self): ...
    def __repr__(self) -> str: ...
    @dataclass
    class ProcessGroupState:
        """
        State for ser-de of process group
        """
        local_rank: int
        global_rank: int
        local_world_size: int
        global_world_size: int
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...

def _create_tensor_from_params(*size, local_device, tensor_properties: TensorProperties):
    """Helper to construct tensor from size, device and common params."""
