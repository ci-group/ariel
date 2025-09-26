import abc
import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from ._internals import check_tensor as check_tensor, get_chunked_dim_size as get_chunked_dim_size, get_split_size as get_split_size, validate_non_overlapping_shards_metadata as validate_non_overlapping_shards_metadata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.distributed._shard.metadata import ShardMetadata as ShardMetadata
from torch.distributed._shard.op_registry_utils import _decorator_func as _decorator_func
from torch.distributed._shard.sharded_tensor import ShardedTensor as ShardedTensor
from typing import Callable

class PlacementSpec(ABC):
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """

@dataclass
class DevicePlacementSpec(PlacementSpec):
    """
    Associates placement of an entity with a single device.

    Args:
        device(:class:`torch.distributed._remote_device`): The device to place the entity on.
    """
    device: torch.distributed._remote_device
    def __post_init__(self) -> None: ...

class ShardingSpec(ABC, metaclass=abc.ABCMeta):
    """
    Base class representing sharding specifications.
    """
    @abstractmethod
    def build_metadata(self, tensor_sizes: torch.Size, tensor_properties: sharded_tensor_meta.TensorProperties) -> sharded_tensor_meta.ShardedTensorMetadata:
        """
        Given a global tensor size, define how to shard a tensor like this shape
        across ranks, return ShardedTensorMetadata
        Args:
            tensor_sizes (:class:`torch.Size`):
                The tensor shape to shard on, a `torch.Size` object that represents the
                tensor shape to be sharded according to the ShardingSpec.
            tensor_properties(:class:`torch.distributed._shard.sharded_tensor.TensorProperties):
                Tensor properties used to create a ShardedTensor.
        Returns:
            A :class:`ShardedTensorMetadata` object that encodes the information about
            the layout of the ShardedTensor and its properties.
        """
    @abstractmethod
    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> ShardedTensor:
        """
        Given a global tensor on src_rank, shard this tensor
        across ranks within the process group, return a ShardedTensor.
        Args:
            tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        Keyword args:
            src_rank (int, optional): The source rank which is used as the ground truth of
                the data for the parameter that would be sharded and scattered
                across the rest of the ranks.
                Default: 0.
            process_group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.
        Returns:
            A :class:`ShardedTensor` sharded from the given tensor.
        """

_CUSTOM_SHARDING_SPEC_OPS: dict[str, dict[Callable, Callable]]

def _has_custom_op(sharding_spec, op):
    """
    Returns whether or not the ShardingSpec has a custom op implementation.
    """
def _dispatch_custom_op(sharding_spec, op: Callable, types, args, kwargs, process_group):
    """
    Calls the custom op for this ShardingSpec if it exists.
    """
def custom_sharding_spec_op(sharding_spec_class, func):
    """
    Decorator to allow custom registration of ops.
    Args:
        sharding_spec_class(type): The ShardingSpec for which we need to add this custom op.
        func(Callable): The op to override (ex: torch.bmm)
    """

@dataclass
class EnumerableShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that allows users to specify a generic
    sharding scheme by enumerating exactly how each shard is laid out.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard. Note that none of the shards should overlap.
    """
    shards: list[ShardMetadata]
    def __post_init__(self) -> None: ...
    def build_metadata(self, tensor_sizes: torch.Size, tensor_properties: sharded_tensor_meta.TensorProperties) -> sharded_tensor_meta.ShardedTensorMetadata: ...
    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> ShardedTensor: ...

def _infer_sharding_spec_from_shards_metadata(shards_metadata):
    """
    Infer the sharding spec from the metadata of each shard of a ShardedTensor.
    If the tensor is sharded only on one dimension, we can then verify whether it's
    a ChunkShardingSpec or not. The way to verify it is to first get the total length
    and perform a chunk sharding with the given placements to see if we can have the
    same chunk size as the given shards_metadata. If not, we assume it's enum sharded.

    Args:
        shards_metadata (List[ShardMetadata]): List of Metadata of local shards.

    Returns:
        A :class:`torch.distributed._shard.sharding_spec.ShardingSpec` object of sharding
            spec for one sharded tensor.
    """
