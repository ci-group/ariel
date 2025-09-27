import torch
from _typeshed import Incomplete
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, get_proxy_slot as get_proxy_slot, track_tensor_tree as track_tensor_tree
from torch.utils._pytree import tree_flatten as tree_flatten
from typing import Any

class ExecutorchCallDelegate(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, lowered_module, *args): ...

executorch_call_delegate: Incomplete
LOWERED_BACKEND_MODULE_TYPE: str

def trace_call_delegate(proxy_mode, func_overload, lowered_module, *args): ...
def call_delegate_cpu(lowered_module, *args): ...
@executorch_call_delegate.py_autograd_impl
def call_delegate_autograd(lowered_module, *args): ...
def call_delegate_proxy_torch_dispatch_mode(mode, lowered_module, *args): ...
def call_delegate_fake_tensor_mode(mode, lowered_module, *args): ...
@executorch_call_delegate.py_functionalize_impl
def call_delegate_functionalize(ctx, lowered_module, *args): ...
def is_lowered_module(obj: Any) -> bool:
    """
    This function is added to avoid using isinstance(obj,
    LoweredBackendModule) as it will import LoweredBackendModule, which may
    cause a circular import.
    """
def get_lowered_module_name(root: torch.nn.Module, lowered_module: LOWERED_BACKEND_MODULE_TYPE) -> str:
    """
    Adds the given lowered_module into the given root module and returns the
    name of the module added.
    """
