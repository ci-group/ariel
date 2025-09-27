import torch
from torch.distributed.tensor import DeviceMesh

__all__ = ['input_reshard']

def input_reshard(module: torch.nn.Module, tp_device_mesh: DeviceMesh, input_reshard_dim: int | None = None) -> torch.nn.Module:
    """
    Register hooks to an nn.Module for input resharding, enabling sharding and restoration during backward computation.

    Register hooks to an nn.Module with input resharding so that we can shard
    per the given `tp_device_mesh` and `input_reshard_dim` and restore the
    input back when recomputing the activations in the backward. The reason
    why we can do this is that for Tensor Parallel(TP), the input are same
    across all TP ranks.

    Args:
        module (:class:`nn.Module`):
            Module to be registered with input resharding.
        tp_device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for Tensor Parallel.
        input_reshard_dim (Optional[int]):
            The dimension of where we perform the sharding
            of input. If set None, there is no sharding of input.
            Default: None

    Return:
        A :class:`nn.Module` object registered with TP input resharding.
    """
