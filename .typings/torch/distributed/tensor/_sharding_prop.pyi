import threading
from _typeshed import Incomplete
from collections.abc import Sequence
from functools import lru_cache
from torch._ops import OpOverload as OpOverload
from torch._subclasses import FakeTensorMode as FakeTensorMode
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec, TensorMeta as TensorMeta
from torch.distributed.tensor._op_schema import OpInfo as OpInfo, OpSchema as OpSchema, OpSpec as OpSpec, OpStrategy as OpStrategy, OutputSharding as OutputSharding, OutputSpecType as OutputSpecType, RuntimeSchemaInfo as RuntimeSchemaInfo, StrategyType as StrategyType, TupleStrategy as TupleStrategy
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset as compute_local_shape_and_global_offset, compute_local_stride as compute_local_stride
from typing import Callable

aten: Incomplete

def _length(obj) -> int: ...

class LocalLRUCache(threading.local):
    cache: Incomplete
    def __init__(self, user_function: Callable) -> None: ...
    def __call__(self, *args, **kwargs) -> object: ...
    def cache_info(self): ...

class ShardingPropagator:
    op_to_rules: dict[OpOverload, Callable[[OpSchema], OutputSharding]]
    op_strategy_funcs: dict[OpOverload, Callable[[OpSchema], StrategyType]]
    op_to_schema_info: dict[OpOverload, RuntimeSchemaInfo]
    propagate_op_sharding: Incomplete
    op_to_shape_and_stride_idx: dict[OpOverload, int | tuple[int, int]]
    def __init__(self) -> None: ...
    def register_sharding_prop_rule(self, op_overload: OpOverload, rule_func: Callable[[OpSchema], OutputSharding], schema_info: RuntimeSchemaInfo | None = None):
        """
        Register a sharding propagation rule for an operator.
        """
    def register_op_strategy(self, op_overload: OpOverload, strategy_func: Callable[[OpSchema], StrategyType], schema_info: RuntimeSchemaInfo | None = None):
        """
        Register a sharding strategy generator for an operator.
        """
    def _propagate_tensor_meta_non_cached(self, op_schema: OpSchema) -> None | TensorMeta | Sequence[TensorMeta | None]:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas
        """
    @lru_cache
    def _propagate_tensor_meta(self, op_schema: OpSchema) -> None | TensorMeta | Sequence[TensorMeta | None]: ...
    def _wrap_output_spec_tensor_meta(self, op: OpOverload, output_specs: OutputSpecType, output_tensor_meta: None | TensorMeta | Sequence[TensorMeta | None]) -> None:
        """
        Wrap the output_specs with the tensor metadata from the output.
        """
    def _wrap_with_op_strategy(self, op_schema: OpSchema) -> OpSchema:
        """
        wrap a op_schema that contains DTensorSpec to another op_schema that contains
        OpStrategy/TupleStrategy, the returned op_schema is then used for sharding
        strategy propagation on pytorch operators.
        """
    def propagate(self, op_info: OpInfo) -> None: ...
    def propagate_op_sharding_non_cached(self, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
    def _select_strategy(self, strategy: OpStrategy) -> OpSpec: ...
    def _adjust_shape_and_stride_args(self, out_tensor_meta: TensorMeta, schema: OpSchema, spec: DTensorSpec) -> OpSchema: ...
