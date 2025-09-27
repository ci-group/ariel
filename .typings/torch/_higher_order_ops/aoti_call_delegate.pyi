import torch
from _typeshed import Incomplete
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor, FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, track_tensor_tree as track_tensor_tree

AOTI_LOWERED_MODULE: str

class AOTICallDelegate(HigherOrderOperator):
    """aoti_call_delegate is a HOP for calling AOTInductor lowered submodule in ExportedProgram.

    It has the following signature:
    aoti_call_delegate(
        lowered_module: Union[AOTInductorEPModule, AOTInductorRunnerWrapper]
        original_gm:fx.GraphModule,
        weight_args: List[Tensor],
        input_args: List[Tensor],
    ) -> outputs: List[Tensor]

    where,
    - lowered_module is the AOTInductor lowered submodule, backed by compiled .so file, supporting real tensor inputs
    - original_gm is the stateless version of the original GraphModule before lowering, allowing FakeTensor propagation
    - weight_args is the list of weights in original GraphModule, including parameters and buffers
    - input_args is the list of flatten inputs
    """
    def __init__(self) -> None: ...
    def __call__(self, lowered_module: AOTI_LOWERED_MODULE, original_gm: torch.fx.GraphModule, weight_args: list[torch.Tensor], input_args: list[torch.Tensor]) -> list[torch.Tensor]: ...

aoti_call_delegate: Incomplete

def call_delegate_cpu(lowered_module: AOTI_LOWERED_MODULE, original_gm: torch.fx.GraphModule, weight_args: list[torch.Tensor], input_args: list[torch.Tensor]) -> list[torch.Tensor]: ...
def trace_aoti_call_delegate(proxy_mode, func_overload, lowered_module, original_gm, weight_args, input_args): ...
def call_delegate_proxy_torch_dispatch_mode(mode: ProxyTorchDispatchMode, lowered_module: AOTI_LOWERED_MODULE, original_gm: torch.fx.GraphModule, weight_args: list[torch.Tensor], input_args: list[torch.Tensor]): ...
def call_delegate_fake_tensor_mode(mode: FakeTensorMode, lowered_module: AOTI_LOWERED_MODULE, original_gm: torch.fx.GraphModule, weight_args: list[torch.Tensor], input_args: list[torch.Tensor]) -> list[torch.Tensor]: ...
@aoti_call_delegate.py_functionalize_impl
def call_delegate_functionalize(ctx, lowered_module: AOTI_LOWERED_MODULE, original_gm: torch.fx.GraphModule, weight_args: list[torch.Tensor], input_args: list[torch.Tensor]): ...
