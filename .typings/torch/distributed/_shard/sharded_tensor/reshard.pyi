import torch
import torch.distributed._shard.sharding_spec as shard_spec
from .shard import Shard as Shard
from torch._C._distributed_c10d import ProcessGroup as ProcessGroup
from torch.distributed._shard.metadata import ShardMetadata as ShardMetadata
from torch.distributed._shard.sharding_spec._internals import get_chunked_dim_size as get_chunked_dim_size, get_split_size as get_split_size
from torch.distributed.nn.functional import all_to_all as all_to_all, all_to_all_single as all_to_all_single

def get_idx_from_placements(placements, current_rank) -> int:
    """
    Return the position of the current rank in the given placements.

    Args:
        placements(List[Union[_remote_device, str]]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This could
            be a list of
            :class:`torch.distributed._remote_device`'s. This list
            could also contain a string which represents remote
            device as accepted by
            :class:`torch.distributed._remote_device`
        current_rank (int): number of current device.

    Returns:
        A int which contains the position of current device in the placement list.
    """
def build_reshard_metadata(st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, world_size: int) -> tuple[list[ShardMetadata], list[int]]:
    """
    Based the given sharding spec, we calculate the offset and local shard size.
    We then build a ShardMetadata on top of the calculation result.

    Args:
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded.
        world_size (int): number of ranks.

    Returns:
        A Tuple of the followings:
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
            A List[int] which contains the ranks in the order of placement.
    """
def reshuffle_local_shard(local_shard: torch.Tensor, st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, resharding_spec: shard_spec.ShardingSpec, pg: ProcessGroup) -> tuple[list[Shard], list[ShardMetadata]]:
    """
    Reshuffle the local shard directly when the reshard dim is same as the original
    sharding dim. Logically we do this in two step:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending the local tensor to
    the new shard directly based on the resharding spec.

    Args:
        local_shard (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the followings:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
def reshard_local_shard(local_tensor: torch.Tensor, st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, resharding_spec: shard_spec.ShardingSpec, pg: ProcessGroup) -> tuple[list[Shard], list[ShardMetadata]]:
    """
    Reshard a sharded tensor given the ``resharding_spec``. When the reshard dim is
    different from the original sharding dim, we need to do two steps logically:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending each rank the new
    shard based on the resharding spec.

    Args:
        local_tensor (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the followings:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
