from _typeshed import Incomplete
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec, TensorMeta as TensorMeta
from torch.distributed.tensor._op_schema import OpSchema as OpSchema, OutputSharding as OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule as register_prop_rule

aten: Incomplete

def convolution_rules(op_schema: OpSchema) -> OutputSharding: ...
def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding: ...
