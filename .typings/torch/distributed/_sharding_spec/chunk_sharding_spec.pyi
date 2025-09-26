import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from ._internals import get_chunked_dim_size as get_chunked_dim_size, get_split_size as get_split_size
from .api import ShardingSpec as ShardingSpec
from dataclasses import dataclass
from torch.distributed._shard._utils import narrow_tensor as narrow_tensor
from torch.distributed._shard.metadata import ShardMetadata as ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor as ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard as Shard
from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device as _parse_and_validate_remote_device

@dataclass
class ChunkShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that defines the placement as being sharded
    across multiple devices. In particular, it represents sharding a Tensor
    along a single dimension into equal chunks (similar to :meth:`torch.chunk`).

    The semantics of how a tensor is partitioned is inline with
    :meth:`torch.chunk`, where ``dim`` in torch.chunk corresponds to the
    specified ``dim`` and ``chunks`` in torch.chunk is the number of elements
    in the placement specified.

    Args:
        dim (int or str):
            The dimension to shard on, could be an integer representing the
            dimension or a string in case of named tensors where dimensions are
            named. Note that named tensor support is not added yet.
        placement(List[Union[_remote_device, str]]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This could
            be a list of
            :class:`torch.distributed._remote_device`'s. This list
            could also contain a string which represents remote
            device as accepted by
            :class:`torch.distributed._remote_device`
    """
    ShardingDim = int | str
    dim: ShardingDim
    placements: list[torch.distributed._remote_device | str]
    def __post_init__(self) -> None: ...
    @staticmethod
    def _verify_dim(dim) -> None: ...
    def build_metadata(self, tensor_sizes: torch.Size, tensor_properties: sharded_tensor_meta.TensorProperties) -> sharded_tensor_meta.ShardedTensorMetadata: ...
    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> ShardedTensor:
        """
        Args:
            src_rank: group rank relative to ``process_group``

            N.B. If ``process_group`` is None, ``src_rank`` is a global rank.
        """
