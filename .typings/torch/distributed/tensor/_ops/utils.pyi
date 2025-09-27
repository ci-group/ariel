import torch
from collections.abc import Iterable, Sequence
from torch._prims_common import DimsSequenceType as DimsSequenceType, DimsType as DimsType
from torch.distributed.tensor._api import DTensor as DTensor
from torch.distributed.tensor._collective_utils import redistribute_cost as redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OpSpec as OpSpec, OpStrategy as OpStrategy, OutputSharding as OutputSharding, PlacementList as PlacementList, RuntimeSchemaInfo as RuntimeSchemaInfo
from torch.distributed.tensor.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

_T = TypeVar('_T')
_P = ParamSpec('_P')

def register_prop_rule(op: torch._ops.OpOverload | list[torch._ops.OpOverload], schema_info: RuntimeSchemaInfo | None = None) -> Callable[[Callable[[OpSchema], OutputSharding]], Callable[[OpSchema], OutputSharding]]: ...
def register_op_strategy(op, schema_info=None) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def as_list(x: list[object] | object) -> list[object] | torch.fx.immutable_collections.immutable_list: ...
def normalize_dim(dim: int, ndim: int) -> int: ...
def normalize_dims(dims: DimsType, ndim: int) -> DimsSequenceType:
    """Normalize a dim or a sequence of dims, so that they are all positive."""
def prod(xs: Iterable[int]) -> int: ...
def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is shardable according to the spec."""
def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is evenly shardable according to the spec."""
def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is sharded."""
def is_tensor_partial(spec: DTensorSpec) -> bool:
    """Return True if tensor is partial on the mesh."""
def infer_broadcast_dims_map(common_shape: torch.Size, input_shape: torch.Size) -> list[int]: ...
def map_placements_after_broadcast(placements: tuple[Placement, ...], shape: torch.Size, broadcast_dims_map: list[int]) -> tuple[Placement, ...]:
    """Map each placement based on the output shape after broadcast."""
def generate_redistribute_costs(src_strategy: OpStrategy, dst_spec: DTensorSpec) -> list[float]: ...
def expand_to_full_mesh_op_strategy(mesh: DeviceMesh, op_schema: OpSchema, single_mesh_dim_strategies: list[PlacementList], *, input_index: int = 1, inplace_op: bool = False) -> OpStrategy: ...
