import torch
from dataclasses import dataclass
from torch.distributed.device_mesh import DeviceMesh

__all__ = ['Placement', 'Shard', 'Replicate', 'Partial']

class Placement:
    """
    The base class for the Placement type, where it describes how a DTensor is placed onto the
    ``DeviceMesh``. ``Placement`` and ``DeviceMesh`` together could describe the DTensor Layout.
    It is the base class of the three main DTensor Placement types: ``Shard``, ``Replicate``,
    and ``Partial``.

    This class is not meant to be used directly, mainly served as a typing stub.
    """
    def is_shard(self, dim: int | None = None) -> bool: ...
    def is_replicate(self) -> bool: ...
    def is_partial(self) -> bool: ...

@dataclass(frozen=True)
class Shard(Placement):
    """
    The ``Shard(dim)`` placement describes the DTensor sharding on tensor dimension
    ``dim`` over a corresponding ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension only holds a shard/piece of the global Tensor. The
    ``Shard(dim)`` placement follows the ``torch.chunk(dim)`` semantic, where the
    last few shards on the DeviceMesh dimension might be empty when the tensor dimension
    is not evenly divisible on the DeviceMesh dimension. The ``Shard`` placement can be
    used by all DTensor APIs (i.e. distribute_tensor, from_local, etc.)

    Args:
        dim (int): The tensor dimension that describes the DTensor is sharded over its
            corresponding DeviceMesh dimension.

    .. warning:: sharding on a tensor dimension where the tensor dimension size is not
        evenly divisible on a DeviceMesh dimension is currently experimental and subject to change.
    """
    dim: int
    def _split_tensor(self, tensor: torch.Tensor, num_chunks: int, *, with_padding: bool = True, contiguous: bool = True) -> tuple[list[torch.Tensor], list[int]]:
        """
        This function uses torch.chunk to split a tensor into num_chunks shards along
        the Shard placement dimension, and return a list of shards with their pad sizes.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
            few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
            This is because collectives usually require equal size tensor inputs
        """
    @staticmethod
    def _local_shard_size_and_offset(curr_local_size: int, num_chunks: int, rank: int) -> tuple[int, int]:
        """
        Given the size of the current local tensor (which may already be sharded on some dimensions),
        computes the new local shard size and offset given the desired number of chunks
        (num_chunks is generally equal to the size of the current sharding dim).

        Note: new local shard offset is relative to the current sharded tensor, not the global tensor.
        See `_utils.compute_local_shape_and_global_offset` for computing global offset.

        Returns (new local shard size, offset)

        """
    def _shard_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, src_data_rank: int | None = 0) -> torch.Tensor:
        """
        shard and scatter a tensor on a mesh dimension (use coordinate
        0 on the mesh dimension as source of truth)
        """
    def _reduce_shard_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, reduce_op: str, mesh_dim: int) -> torch.Tensor:
        """
        reduce and scatter a tensor on a mesh dimension
        """
    def _to_replicate_tensor(self, local_tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, current_logical_shape: list[int]) -> torch.Tensor:
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
    def _replicate_to_shard(self, local_tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, shard_index: int) -> torch.Tensor:
        """
        transform from replicated tensor to a sharded tensor on
        the current rank, which would perform a local chunk
        """
    def _to_new_shard_dim(self, local_tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, current_logical_shape: list[int], new_shard_dim: int) -> torch.Tensor:
        """
        transform from existing sharded tensor to a new sharded tensor on
        that shard on a new dimension, which performs an alltoall
        """
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str:
        """
        machine readable representation of the Shard placement
        """
    def __str__(self) -> str:
        """human readable representation of the Shard placement"""

@dataclass(frozen=True, **kw_only_dataclass)
class _StridedShard(Shard):
    '''
    _StridedShard is only introduced to support 2D FSDP2 + TP sharding where the tensor
    is sharded on the TP mesh dimension first, then sharded on the FSDP mesh dimension.
    We call this right-to-left sharding which is the opposite of the default
    left-to-right sharding. See the example below:
        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [Shard(0), Shard(0)]

    The default sharding behavior shards the tensor on "dp" mesh dimension first then
    "tp" dimension. The sharding result will be:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   1 (row 2-3)
        2       |   (1, 0)          |   2 (row 4-5)
        3       |   (1, 1)          |   3 (row 6-7)

    While the FSDP2 + TP sharding behavior does the opposite: it shards the tensor on
    "tp" mesh dim first then "dp" dim. This right-to-left sharding will produce the
    result:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The consequence is, any attempt to redistribute this DTensor to a full replica will
    produce a wrong result because the shard-to-replicate redistribution always happens
    right-to-left, regardless it\'s left-to-right sharding or right-to-left. To address
    this, we use _StridedShard placement to make this right-to-left sharding compatible
    with our left-to-right convention on both tensor distribution and redistribution.

    Now with _StridedShard, the right-to-left sharding above can be represented as:
        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [_StridedShard(0, split_factor=2), Shard(0)]

    And a left-to-right processing of `placements` will produce the same result, which is
    different from using the `Shard` placement:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The argument `split_factor` is the number of existing shards over the tensor sharding
    dimension before processing the _StridedShard placement, as if the sharding happened
    right-to-left. In the example above, the tensor should first be sharded on the "tp"
    dimension into 2 shards before being sharded on the "dp" dimension. Therefore, the
    `split_factor` of the _StridedShard placement on "dp" dim is 2.

    TODO: we should remove _StridedShard placement once we can unify it with Shard
    '''
    split_factor: int
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str:
        """
        machine readable representation of the _StridedShard placement
        """
    def __str__(self) -> str:
        """human readable representation of the _StridedShard placement"""
    def _split_tensor(self, tensor: torch.Tensor, num_chunks: int, *, with_padding: bool = True, contiguous: bool = True) -> tuple[list[torch.Tensor], list[int]]: ...
    def _to_replicate_tensor(self, local_tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, current_logical_shape: list[int]) -> torch.Tensor:
        '''
        Given a tensor with strided sharding (e.g. [StridedShard(d), Shard(d)]),
        this function is called during the process of converting to [Replicate(), Replicate()],
        and `local_tensor` represents the portion of the tensor on this rank after the intermediate step of
        converting to [StridedShard(d), Replicate()] in right-to-left unsharding order.

        note: this conversion logic is pretty specialized on this 2D case.  It could be generalized further. This
        is a common enough case to be worth fixing (since it occurs when applying TP and then FSDP to a model).

        note: this does not support \'reduce_scatter\' for StridedShard.

        Example
        -------
        mesh = (DP=2, TP=2)
        # single-gpu "weight" of size 5, will be \'uneven\' for sharding
        original = torch.arange(5)

        tp sharded tensor
        -----------------
        `tp = distribute_tensor(x, world_mesh[\'tp\'], [Shard(0)])`

        local_tensors:
        rank0: [0,1,2]    rank1: [3,4]
        rank1: [0,1,2]    rank3: [3,4]

        fsdp+tp sharded tensor
        ----------------------
        `dp_tp = ...` (the process of creating a strided-shard tensor is skipped over as it is complicated
        dp_tp has placement (_StridedShard(0, split_factor=2), Shard(0))
        local_tensors:
        rank0: [0,1]  rank1: [3]
        rank1: [2]    rank3: [4]

        Now, say someone wants to reconstruct dp_tp\'s full tensor. This will invoke \'redistribute\' to replicate.
        redistribute will first replicate the "Shard(0)" placement on the rightmost mesh dim, then replicate the
        StridedShard placement second, which is implemented by this function.
        So our starting point (`local_tensor` arg) is the result of replicating the Shard(0) placement across the
        TP dim, which looks like this.

        Note the discrepancy with the \'tp sharded tensor\' line above!  We\'ll fix it by locally shuffling data.

        local_tensors:
        rank0: [0,1,3]  rank1: [0,1,3]
        rank2: [2,4]    rank3: [2,4]

        Step 1: replicate over the DP dimension.  Afterwards, each rank can locally sort the values.
          note: we need padding to do this allgather, and we\'ll need to keep track of the padding amount for later
                local_tensors:
        rank0: [0,1,3,2,4]    rank1: [0,1,3,2,4]
        rank2: [0,1,3,2,4]    rank3: [0,1,3,2,4]

        Step 2: chunk and shuffle values around to account for the wrong order of operations above
        and get the original tensor content back

        01324#       <- our allgather includes padding, if padding was applied in step 1
        01324        <- Remove the padding
        013, 24      <- chunk once, \'undoing\' the DP allgather
        01, 3, 2, 4  <- chunk each chunk, \'undoing\' the initial (wrong) TP allgather performed by Shard(0)->Replicate()
        012, 34      <- interleave with stride=TP mesh dim size
        01234        <- concatenate

        Note: the current implementation of this function is incomplete, and supports only the common pattern of one
        strided shard placement, which is used in the FSDP + TP case.  We could extend this implementation to handle
        multiple strided shardings (e.g. [StridedShard, StridedShard, Shard]), by repeating the chunking step more times
        and handling more complex shuffling in the last step.  On the other hand, we plan to replace \'StridedShard\'
        with using just Shard and specifying a sharding order, so it may be ok to leave this as-is for the time being.
        '''

@dataclass(frozen=True)
class Replicate(Placement):
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. ``distribute_tensor``, ``DTensor.from_local``, etc.)
    """
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str:
        """
        machine readable representation of the Replicate placement
        """
    def __str__(self) -> str:
        """
        human readable representation of the Replicate placement
        """
    def _replicate_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, src_data_rank: int | None = 0) -> torch.Tensor:
        """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """

@dataclass(frozen=True)
class Partial(Placement):
    '''
    The ``Partial(reduce_op)`` placement describes the DTensor that is pending
    reduction on a specified ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension holds the partial value of the global Tensor. User can
    redistribute the ``Partial`` DTensor to a ``Replicate`` or ``Shard(dim)``
    placement on the specified ``DeviceMesh`` dimension using ``redistribute``,
    which would trigger necessary communication operations under the hood (i.e.
    ``allreduce``, ``reduce_scatter``).

    Args:
        reduce_op (str, optional): The reduction op to be used for the partial DTensor
            to produce Replicated/Sharded DTensor. Only element-wise reduction operations
            are supported, including: "sum", "avg", "product", "max", "min", default: "sum".

    .. note:: The ``Partial`` placement can be generated as a result of the DTensor operators,
        and can only be used by the ``DTensor.from_local`` API.
    '''
    reduce_op: str = ...
    def _reduce_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor: ...
    def _reduce_shard_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, shard_spec: Placement) -> torch.Tensor: ...
    def _partition_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
    def __str__(self) -> str:
        """
        human readable representation of the Partial placement
        """
_Partial = Partial
