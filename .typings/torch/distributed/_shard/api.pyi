import torch
import torch.distributed as dist
import torch.nn as nn
from .sharder import Sharder as Sharder
from .sharding_plan import ShardingPlan as ShardingPlan
from .sharding_spec import ChunkShardingSpec as ChunkShardingSpec, ShardingSpec as ShardingSpec
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from torch.distributed import distributed_c10d as distributed_c10d
from torch.distributed._shard.sharded_tensor import ShardedTensor as ShardedTensor

def _shard_tensor(tensor: torch.Tensor, sharding_spec: ShardingSpec, src_rank: int = 0, process_group=None) -> ShardedTensor:
    """
    Given a :class:`torch.Tensor`, it shards that tensor according to the provided
    ``sharding_spec``. ``src_rank`` denotes the source rank which would be
    used as the ground truth of the data which would be scattered as shards
    across the rest of the ranks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        A :class:`ShardedTensor` sharded from the given tensor.

    .. warning::
        Only :class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec` is
        currently supported as the ``sharding_spec``.
    """
def shard_parameter(module: torch.nn.Module, param_name: str, sharding_spec: ShardingSpec, src_rank: int = 0, process_group=None):
    """
    Given a :class:`torch.nn.Module`, a ``param_name`` for a parameter in that
    module, it shards that parameter according to the provided
    ``sharding_spec``. ``src_rank`` denotes the source rank which would be
    used as the ground truth of the data which would be scattered as shards
    across the rest of the ranks.

    This method replaces ``module.param_name`` with a
    :class:`torch.distributed._sharded_tensor.ShardedTensor`

    Args:
        module (:class:`torch.nn.Module`): Module whose parameter needs to be sharded.
        param_name (str): Name of the parameter of ``module`` that needs to be sharded.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    .. warning::
        Only :class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec` is
        currently supported as the ``sharding_spec``.
    """

_CURRENT_PROCESS_GROUP: dist.ProcessGroup | None

@contextmanager
def load_with_process_group(process_group) -> Generator[Incomplete]:
    """
    Context manager to set the process group with which to load a ShardedTensor.
    """
def _get_current_process_group():
    """
    Retrieves the current process group set by ``load_with_process_group``.
    If not set, it just returns the default group.
    """
def _reshard_output(module: torch.nn.Module, resharding_spec: ShardingSpec) -> torch.nn.Module:
    """
    Hook a module with output resharding in the forward pass according
    to the given ``resharding_spec``.

    Args:
        module (:class:`torch.nn.Module`): Module whose output needs to be resharded.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            The specification describing how the output of the module will be resharded.

    Returns:
        A :class:`torch.nn.Module` object with reshard API hooked.
    """
def _collect_local_shard(module: torch.nn.Module) -> torch.nn.Module:
    """
    Hook a module with local shards collection in the forward pass.

    This API is typically used to convert a sharded representation back to data parallel
    representation. In particular, it returns the local tensor for this Shard. If the
    size along the sharding dimension for the local tensor is 1, this dimension is removed
    from the final result. For example a [4, 16] ShardedTensor across 4 ranks is typically
    a local Tensor of size [16] across each rank and not [1, 16] across each rank.

    Args:
        module (:class:`torch.nn.Module`): Module whose output is ShardedTensor and the
            local tensor value needs to be returned.

    Returns:
        A :class:`torch.nn.Module` object with collection API hooked.
    """
def shard_module(module: nn.Module, plan: ShardingPlan, src_rank: int = 0, process_group=None):
    """
    Shards a given module according to the provided sharding `plan`. This method
    first shards all the parameters according to the given sharding `plan`. Then if
    `output_plan` and `return_local_tensor` are specified in the sharding `plan`, it
    will tag the output of modules according `output_plan`, convert the module's
    output back to data parallel according to `return_local_tensor`.

    Needs to be called on all ranks in an SPMD fashion.

    Args:
        module (:class:`torch.nn.Module`): The module to apply sharding to
        plan (:class:`torch.distributed._shard.sharding_plan.ShardingPlan`):
            The ShardingPlan which specified param name to ShardingSpec to apply to
            each parameter.

    Keyword args:
         src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the module that would be sharded and scattered across the rest
            of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    """
