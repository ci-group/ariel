import torch
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from torch._C import DispatchKey as DispatchKey
from torch._functorch._aot_autograd.utils import KNOWN_TYPES as KNOWN_TYPES
from torch._higher_order_ops.utils import autograd_not_implemented as autograd_not_implemented
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject, _is_script_object as _is_script_object, _ns_and_class_name as _ns_and_class_name
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, track_tensor_tree as track_tensor_tree
from torch.fx.node import has_side_effect as has_side_effect

log: Incomplete

class CallTorchBind(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, obj, method, *args, **kwargs): ...
    @staticmethod
    def schema(obj, method) -> torch.FunctionSchema:
        """
        Returns the schema of ``CallTorchbind.__call__``.
        """

call_torchbind: Incomplete
_orig_scriptmethod_call: Incomplete

def torchbind_method_redispatch(self, *args, **kwargs): ...
@contextmanager
def enable_torchbind_tracing() -> Generator[None]:
    """Context manager that acts as a feature flag to enable torchbind tracing
    behavior. Once torchbind tracing has been stabilized, we can remove this and
    turn it always on.
    """
def call_torchbind_impl(obj, method, *args, **kwargs): ...
def inner(mode, *args, **kwargs): ...
def call_torchbind_fake(mode, *args, **kwargs): ...
@call_torchbind.py_functionalize_impl
def call_torchbind_func(ctx, *args, **kwargs): ...
