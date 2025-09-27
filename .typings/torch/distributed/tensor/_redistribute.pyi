import torch
import torch.distributed.tensor._api as dtensor
from _typeshed import Incomplete
from functools import cache
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec, TensorMeta as TensorMeta
from torch.distributed.tensor.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard
from typing import NamedTuple

logger: Incomplete

class _TransformInfo(NamedTuple):
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    logical_shape: list[int]

def _gen_transform_infos_non_cached(src_spec: DTensorSpec, dst_spec: DTensorSpec) -> list[_TransformInfo]:
    """
    Generate the transform infos from the source placements to the target placements.

    To transform from source to target placement it might have multiple steps, i.e. it
    might decompose Si -> Sj into Si -> R -> Sj.
    This would detect if there're mis-aligned/nested shardings between src/dst placements.
    E.g. Suppose the redistribution to perform is (Shard(0), Shard(0)) -> (Replicate(), Shard(0)),
    in this case Shard(0) -> Shard(0) for mesh dimension 1 actually needs resharding, because in
    the former is a nested-sharding of a tensor already already sharded dimension 0, whereras
    the latter is the first sharding on tensor dimension 0.
    """
@cache
def _gen_transform_infos(src_spec: DTensorSpec, dst_spec: DTensorSpec) -> list[_TransformInfo]: ...
def redistribute_local_tensor(local_tensor: torch.Tensor, current_spec: DTensorSpec, target_spec: DTensorSpec, *, async_op: bool = False, is_backward: bool = False) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """

class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: dtensor.DTensor, device_mesh: DeviceMesh, placements: tuple[Placement, ...], async_op: bool = False, forward_dtype: torch.dtype | None = None, backward_dtype: torch.dtype | None = None): ...
    @staticmethod
    def backward(ctx, grad_output: dtensor.DTensor): ...
