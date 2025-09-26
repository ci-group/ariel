from _typeshed import Incomplete
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OpSpec as OpSpec, OpStrategy as OpStrategy, StrategyType as StrategyType
from torch.distributed.tensor._ops.utils import is_tensor_partial as is_tensor_partial, register_op_strategy as register_op_strategy

aten: Incomplete

def random_op_strategy(op_schema: OpSchema) -> StrategyType: ...
