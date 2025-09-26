from _typeshed import Incomplete
from collections.abc import Sequence
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OpSpec as OpSpec, OpStrategy as OpStrategy, RuntimeSchemaInfo as RuntimeSchemaInfo, StrategyType as StrategyType, TupleStrategy as TupleStrategy
from torch.distributed.tensor._ops.utils import generate_redistribute_costs as generate_redistribute_costs, infer_broadcast_dims_map as infer_broadcast_dims_map, map_placements_after_broadcast as map_placements_after_broadcast, normalize_dim as normalize_dim, register_op_strategy as register_op_strategy
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard

aten: Incomplete
linear_pointwise_ops: Incomplete
pointwise_ops: Incomplete

def pointwise_strategy(op_schema: OpSchema, linearity: bool = False) -> OpStrategy: ...
def common_pointwise_strategy(args_schema: Sequence[object], followed_strategy: OpStrategy, linearity: bool) -> OpStrategy: ...
def linear_pointwise_strategy(op_schema: OpSchema) -> StrategyType:
    """
    Linear pointwise operators can propagate pending reductions.
    For example, c = add(a, b); if a is pending sum, then c will be
    pending sum as well without any communication overhead.
    """

for_each_ops: Incomplete
for_each_linearity_ops: Incomplete

def list_pointwise_strategy(op_schema: OpSchema, linearity: bool = False) -> StrategyType:
    """
    Apply the pointwise strategy to the zipped arguments. For example, if we
    run a foreach add of two lists l1 and l2, then we apply the pointwise
    strategy on each pair (l1[i], l2[i]). If the first argument is a list but
    the second (or later) one is a tensor, then we broadcast the tensor by
    replicating it into a list with the length of the first argument.

    Args:
        mesh (DeviceMesh): device mesh for pointwise ops
        op_schema (OpSchema): schema of the operator to generate strategy for
        linearity (bool): specify whether op(a) + op(b) = op(a + b)

    Returns:
        OpStrategy: generated strategy
    """
def list_linear_pointwise_strategy(op_schema: OpSchema) -> StrategyType:
    """
    for each list op stratgy that supports linearity
    """

fused_ops: Incomplete
