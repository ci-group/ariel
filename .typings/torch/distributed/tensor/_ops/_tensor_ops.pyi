import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OpSpec as OpSpec, OpStrategy as OpStrategy, OutputSharding as OutputSharding, PlacementList as PlacementList, RuntimeSchemaInfo as RuntimeSchemaInfo, StrategyType as StrategyType, TupleStrategy as TupleStrategy
from torch.distributed.tensor._ops._common_rules import pointwise_rule as pointwise_rule
from torch.distributed.tensor._ops._embedding_ops import _MaskPartial as _MaskPartial
from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy as expand_to_full_mesh_op_strategy, generate_redistribute_costs as generate_redistribute_costs, is_tensor_dim_sharded as is_tensor_dim_sharded, is_tensor_evenly_shardable as is_tensor_evenly_shardable, is_tensor_partial as is_tensor_partial, normalize_dim as normalize_dim, register_op_strategy as register_op_strategy, register_prop_rule as register_prop_rule
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate, Shard as Shard

aten: Incomplete

def default_strategy(op_schema: OpSchema) -> StrategyType: ...
def equal_strategy(op_schema: OpSchema) -> StrategyType: ...
def create_like_strategy(op_schema: OpSchema) -> StrategyType: ...
def new_factory_strategy(op_schema: OpSchema) -> StrategyType: ...
def gen_bucketize_strategy(op_schema: OpSchema) -> StrategyType:
    """Just propagate input sharding, but expect replicated for boundaries input."""
def select_int_strategy(op_schema: OpSchema) -> StrategyType:
    """
    In this select op, first determine the input specs, then determine the output specs.
    - Input specs:
        - If the input is sharded on the selected dim, unshard it and change to replicate.
        - Otherwise, keep the original input specs.
    - Output specs:
        - It checks the input specs with the following cases:
        - Case 1 shard_dim == selected_dim: not possible as the input is already unsharded.
        - Case 2 shard_dim < selected_dim: keep the input specs.
        - Case 3 shard_dim > selected_dim: shard_dim -= 1.
    """
def select_backward_strategy(op_schema: OpSchema) -> OpStrategy: ...
def gen_slice_strategy(op_schema: OpSchema) -> StrategyType:
    """Forward all shardings except the slice dimension."""
def slice_backward_rules(op_schema: OpSchema) -> OpStrategy: ...
def unshard_tensor_dim(placements: Sequence[Placement], dim: int) -> tuple[Placement, ...]:
    """Disallow the given tensor dimension to be sharded."""
def replicate_tensor_dim(placements: Sequence[Placement], dim: int) -> tuple[Placement, ...]:
    """Force the given tensor dimension to be replicated."""
def gen_slice_scatter_strategy(op_schema: OpSchema) -> StrategyType: ...
def replica_only_strategy(op_schema: OpSchema) -> StrategyType:
    """Only allow replication on the input/output."""
def scatter_strategy(op_schema: OpSchema) -> StrategyType: ...
def gather_strategy(op_schema: OpSchema) -> StrategyType: ...
def _derive_follow_placements_from_tuple_strategy(op: torch._ops.OpOverload, tuple_strategy: TupleStrategy) -> Sequence[Placement]:
    """
    derive the placements to follow from the tuple strategy, mainly used by
    aten.stack, aten.cat, where each operand have the same shape, and correspondingly
    expecting the same sharding
    """
def normalize_shard_for_stack(placements: Sequence[Placement], insert_dim: int = 0) -> Sequence[Placement]: ...
def stack_strategy(op_schema: OpSchema) -> StrategyType: ...
def cat_strategy(op_schema: OpSchema) -> StrategyType: ...
def prop_index_select(op_schema: OpSchema) -> OutputSharding: ...
def prop_index(op_schema: OpSchema) -> OutputSharding:
    '''
    Expect replicated on the first input; _mostly_ pointwise on the second input.

    TODO: exception: when the dtype of second input is "bool", then a torch.nonzero needs to be triggered first.
    '''
def split_strategy(op_schema: OpSchema) -> TupleStrategy: ...
