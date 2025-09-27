from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.flat_apply import _ConstantFunction as _ConstantFunction, flat_apply as flat_apply, to_graphable as to_graphable
from torch._higher_order_ops.strict_mode import strict_mode as strict_mode
from torch._higher_order_ops.utils import autograd_not_implemented as autograd_not_implemented
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import PreDispatchTorchFunctionMode as PreDispatchTorchFunctionMode, ProxyTorchDispatchMode as ProxyTorchDispatchMode, get_proxy_slot as get_proxy_slot, track_tensor_tree as track_tensor_tree
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type as is_traceable_wrapper_subclass_type

class ExportTracepoint(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, *args, **kwargs): ...

_export_tracepoint: Incomplete

def export_tracepoint_dispatch_mode(mode, *args, **kwargs): ...
def export_tracepoint_fake_tensor_mode(mode, *args, **kwargs): ...
@_export_tracepoint.py_functionalize_impl
def export_tracepoint_functional(ctx, *args, **kwargs): ...
def export_tracepoint_cpu(*args, **kwargs): ...
def _wrap_submodule(mod, path, module_call_specs): ...
@contextmanager
def _wrap_submodules(f, preserve_signature, module_call_signatures) -> Generator[None]: ...
def _mark_strict_experimental(cls): ...
def _register_subclass_spec_proxy_in_tracer(tracer, name, spec):
    """
    This is a wrapper utility method on top of tracer to cache the
    already registered subclass spec attribute. This is useful because
    Subclass.__init__ will be same for each subclass. By default, fx will
    create multiple attributes/proxies for given attribute.
    """
def mark_subclass_constructor_exportable_experimental(constructor_subclass):
    """
    Experimental decorator that makes subclass to be traceable in export
    with pre-dispatch IR. To make your subclass traceble in export, you need to:
        1. Implement __init__ method for your subclass (Look at DTensor implementation)
        2. Decorate your __init__ method with _mark_constructor_exportable_experimental
        3. Put torch._dynamo_disable decorator to prevent dynamo from peeking into its' impl

    Example:

    class FooTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, elem, *, requires_grad=False):
            # ...
            return torch.Tensor._make_subclass(cls, elem, requires_grad=requires_grad)

        @torch._dynamo_disable
        @mark_subclass_constructor_exportable_experimental
        def __init__(self, elem, ...):
            # ...
    """
