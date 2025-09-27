import torch
from _typeshed import Incomplete
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented as autograd_not_implemented
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND as ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes as elementwise_dtypes
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, maybe_handle_decomp as maybe_handle_decomp, track_tensor_tree as track_tensor_tree

ALLOWABLE_OPS: Incomplete

class OutDtypeOperator(HigherOrderOperator):
    """
    The out_dtype operator takes an existing ATen functional operator, an
    `out_dtype` argument, and arguments to the original operator, and executes
    the original operator and returns a Tensor with the `out_dtype` precision.
    This operator does not mandate a compute precision so it allows the
    representation to not be opinionated about the exact implementation.

    The general implementation for all operators will be the following:
        1. Promote inputs dtypes based on default PyTorch dtype promotion rules,
            using the dtypes of all input Tensors/Scalars and the `out_dtype`
            arugument.
        2. Execute the operator
        3. Cast the output to `out_dtype`
    """
    def __init__(self) -> None: ...
    def __call__(self, op, output_dtype, *args): ...

out_dtype: Incomplete

def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args): ...
def out_dtype_dense(op: torch._ops.OpOverload, output_dtype: torch.dtype, *args): ...
def is_int_mm(op, output_dtype, args): ...
def out_dtype_fallback(op, output_dtype, *args): ...
def out_dtype_proxy(mode: ProxyTorchDispatchMode, op: torch._ops.OpOverload, output_dtype: torch.dtype, *args): ...
def out_dtype_fake_tensor_mode(mode: FakeTensorMode, op: torch._ops.OpOverload, output_dtype: torch.dtype, *args): ...
@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args): ...
