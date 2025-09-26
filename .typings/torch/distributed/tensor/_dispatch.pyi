import torch
from _typeshed import Incomplete
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec, TensorMeta as TensorMeta
from torch.distributed.tensor._op_schema import OpInfo as OpInfo, OpSchema as OpSchema, OutputSpecType as OutputSpecType
from torch.distributed.tensor._random import is_rng_supported_mesh as is_rng_supported_mesh
from torch.distributed.tensor._redistribute import redistribute_local_tensor as redistribute_local_tensor
from torch.distributed.tensor._sharding_prop import ShardingPropagator as ShardingPropagator
from torch.distributed.tensor._tp_conv import convolution_backward_handler as convolution_backward_handler, convolution_handler as convolution_handler
from torch.distributed.tensor._utils import try_find_mesh_from_args as try_find_mesh_from_args
from torch.distributed.tensor.placement_types import Partial as Partial, Placement as Placement, Replicate as Replicate

aten: Incomplete
logger: Incomplete

def is_same_size_handler(op_call: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]) -> bool: ...
def found_inf_reduce_handler(op_call: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]) -> None: ...

class OpDispatcher:
    """
    Op dispatching class instance to handle args/kwargs pre-processing (un-wrapping), sharding
    propagation, redistribute local args, local compute, and post-processing (re-wrapping). It
    also handles any op specific logic if necessary.

    NOTE: Given the runtime overhead of Tensor subclass (__torch_dispatch__), the OpDispatcher
    is designed to minimize the CPU overhead by using the tricks of proper unflattening, faster
    pytree if needed, and leveraging various caching mechanisms implemented in the sharding
    propagation and redistribute modules. The CPU overhead is critical to eager mode performance,
    one need to carefully measure the CPU overhead when making significant changes to the
    OpDispatcher and ShardingPropagator.
    """
    sharding_propagator: Incomplete
    _random_ops: Incomplete
    _custom_op_handlers: Incomplete
    _allow_implicit_replication: bool
    def __init__(self) -> None: ...
    def dispatch(self, op_call: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        """
        Main dispatching logic
        """
    @staticmethod
    def redistribute_local_args(op_info: OpInfo, suggested_input_schema: OpSchema) -> None: ...
    def unwrap_to_op_info(self, op_call: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]) -> OpInfo: ...
    @staticmethod
    def wrap(res: object, spec: OutputSpecType) -> object: ...
    def _try_replicate_spec_for_scalar_tensor(self, op_call: torch._ops.OpOverload, tensor_arg: torch.Tensor, compute_mesh: DeviceMesh) -> DTensorSpec: ...
