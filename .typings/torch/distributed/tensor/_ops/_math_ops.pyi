import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OpSpec as OpSpec, OpStrategy as OpStrategy, PlacementList as PlacementList, RuntimeSchemaInfo as RuntimeSchemaInfo, TupleStrategy as TupleStrategy
from torch.distributed.tensor._ops.utils import as_list as as_list, expand_to_full_mesh_op_strategy as expand_to_full_mesh_op_strategy, generate_redistribute_costs as generate_redistribute_costs, is_tensor_evenly_shardable as is_tensor_evenly_shardable, normalize_dim as normalize_dim, normalize_dims as normalize_dims, register_op_strategy as register_op_strategy
from torch.distributed.tensor._utils import normalize_to_torch_size as normalize_to_torch_size
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard

aten: Incomplete

class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2

@dataclass(frozen=True)
class NormReduction:
    norm_type: int | float | str
ReductionOpType = NormReduction | str

@dataclass(frozen=True)
class _NormPartial(Partial):
    """
    This placement is used for partial vector norm.

    For p-norms (where p not inf or -inf), the p-norm over n elements computes
        (sum_i x_i^p)^(1/p)
    where the sum is from i=1 to n. The reduction op is the p-norm itself.
    For example, consider 2 ranks, a (4,) tensor sharded on dim-0, and 2-norm:
        Rank 0: [t1, t2] | Rank 1: [t3, t4]
    After computing 2-norm per gradient (partial placement):
        Rank 0: [sqrt(t1^2 + t2^2)] | Rank 1: [sqrt(t3^2 + t4^2)]
    Converting from partial to replicate wants to ultimately get:
        Rank 0/1: [sqrt(t1^2 + t2^2 + t3^2 + t4^2)]
    This can be achieved by computing 2-norm on each rank's result. This holds
    similarly for inf and -inf norm. For 0-norm, the reduction op is sum.
    """
    norm_type: int | float | str = ...
    def __post_init__(self) -> None:
        """Set the appropriate reduce op based on the norm type."""
    def _partition_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        """
        For example, consider 4 ranks, a (3,) replicated tensor, and 2-norm:
            Ranks 0 and 1: sqrt(t1^2 + t2^2 + t3^3)
        To convert from replicated to partial, we want f(x) such that
            sqrt(t1^2 + t2^2 + t3^3) = sqrt(4f(t1)^2 + 4f(t2)^2 + 4f(t3)^2)
                                     = sqrt(4) sqrt(f(t1)^2 + f(t2)^2 + f(t3)^2).
        One such f(x) is f(x) = x / sqrt(4). This generalizes to d ranks and
        p-norm as f(x) = x / d^(1/p).
        """
    def _reduce_shard_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, shard_spec: Placement) -> torch.Tensor: ...
    def _reduce_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor: ...
    def _pre_reduce_transform(self, tensor: torch.Tensor) -> torch.Tensor: ...
    def _post_reduce_transform(self, tensor: torch.Tensor) -> torch.Tensor: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def _infer_reduction_dims(dims_arg: object, ndim: int) -> list[int] | None: ...
def _infer_reduce_dims_map(reduction_dims: list[int], input_ndim: int, keep_dim: bool = False) -> list[int]: ...
def _replicate_dims_start_at(placements: Sequence[Placement], start_dim: int = 0) -> tuple[Placement, ...]: ...
def _skip_dim(placements: tuple[Placement, ...], skipped_dim: int) -> tuple[Placement, ...]: ...
def replicate_reduction_dims(placements: tuple[Placement, ...], reduction_dims: list[int]) -> tuple[Placement, ...]: ...
def map_placements_after_reduction(placements: tuple[Placement, ...], reduction_dims: list[int], reduction_dims_map: list[int], reduction_op: ReductionOpType) -> tuple[Placement, ...]:
    """
    Map each placement based on the output shape after reduction.
    """
def get_placement_from_reduction_op(reduction_op: ReductionOpType) -> Placement: ...
def common_reduction_strategy(input_strategy: OpStrategy, reduce_dims: list[int], keep_dim: bool = False, reduction_linear: bool = True, reduction_op: ReductionOpType = 'sum') -> OpStrategy:
    """
    reduction_linear means that the reduction `f` follows this rule:
        f([f(a), f(b)]) = f([a, b])

    reduction linear should be super set of linearity.
    """

LINEAR_REDUCTION_OP_MAP: Incomplete

def linear_reduction_strategy(op_schema: OpSchema) -> OpStrategy: ...
def cumsum_strategy(op_schema: OpSchema) -> OpStrategy: ...
def var_reduction_strategy(op_schema: OpSchema) -> OpStrategy: ...
def vector_norm_strategy(op_schema: OpSchema) -> OpStrategy: ...
def foreach_norm_strategy(op_schema: OpSchema) -> TupleStrategy: ...
def linalg_replicate_strategy(op_schema: OpSchema) -> OpStrategy:
    """
    Since we do not have a simple way to compute some linear algebra operations
    like SVD or QR decomposition, always fall back to replicate.
    """
def softmax_strategy(op_schema: OpSchema) -> OpStrategy: ...
def softmax_backward_strategy(op_schema: OpSchema) -> OpStrategy: ...
def nll_loss_forward_strategy(op_schema: OpSchema) -> OpStrategy: ...
def nll_loss_backward_strategy(op_schema: OpSchema) -> OpStrategy: ...
def layer_norm_strategy(op_schema: OpSchema) -> OpStrategy: ...
def layer_norm_bwd_strategy(op_schema: OpSchema) -> OpStrategy: ...
def topk_strategy(op_schema: OpSchema) -> OpStrategy: ...
