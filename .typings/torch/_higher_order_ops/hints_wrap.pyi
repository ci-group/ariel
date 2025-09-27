from _typeshed import Incomplete
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented as autograd_not_implemented, reenter_make_fx as reenter_make_fx, unique_graph_id as unique_graph_id
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, track_tensor_tree as track_tensor_tree

class HintsWrapper(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, body_fn, args, kwargs, hints):
        """
        Call implementation of hints_wrapper

        Args:
            body_fn (Callable): A callable function that is within the scope
             that is being traced.

            args (Tuple of torch.Tensor/int/float/bool): A tuple of inputs to
             body_fn.

            kwargs (dict): Keyword argument to the body_fn.

            hints (dict): A dict of context hints which could be passed to
             backend compiler.
        """

hints_wrapper: Incomplete

def hints_wrapper_dense(body_fn, args, kwargs, hints): ...
def hints_wrapper_fake_tensor_mode(mode, body_func, args, kwargs, hints): ...
@hints_wrapper.py_functionalize_impl
def hints_wrapper_functionalize(ctx, body_fn, args, kwargs, hints): ...
def trace_hints_wrapper(proxy_mode, hints_wrapper, body_fn, args, kwargs, hints): ...
def inner(proxy_mode, body_fn, args, kwargs, hints): ...
