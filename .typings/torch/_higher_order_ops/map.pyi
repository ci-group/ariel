import torch
import torch.utils._pytree as pytree
from .utils import _from_fun as _from_fun, _stack_pytree as _stack_pytree, _unstack_pytree as _unstack_pytree, clone_outputs_aliasing_inputs as clone_outputs_aliasing_inputs, prepare_fw_with_masks as prepare_fw_with_masks, save_tensors_and_symints_for_backward as save_tensors_and_symints_for_backward, saved_tensors_and_symints as saved_tensors_and_symints
from _typeshed import Incomplete
from torch._C import DispatchKey as DispatchKey
from torch._dispatch.python import suspend_functionalization as suspend_functionalization
from torch._higher_order_ops.utils import _maybe_run_with_interpreter as _maybe_run_with_interpreter, reenter_make_fx as reenter_make_fx
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode as disable_functional_mode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, make_fx as make_fx, track_tensor_tree as track_tensor_tree
from typing import Callable
from typing_extensions import TypeVarTuple

class MapImpl(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, *args, **kwargs): ...

map_impl: Incomplete

def create_fw_bw_graph(f, num_mapped_args, *args): ...
def map(f: Callable[[pytree.PyTree, tuple[pytree.PyTree, ...]], pytree.PyTree], xs: pytree.PyTree | torch.Tensor, *args: TypeVarTuple):
    """
    Perfoms a map of f with xs. Intuitively, you can think of the semantic being:

    out = []
    for idx in len(xs.size(0)):
        xs_sliced = xs.select(0, idx)
        out.append(f(xs_sliced, *args))
    torch.stack(out)

    .. warning::
        `torch._higher_order_ops.map` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype


    Args:
        f (Callable): a callable that takes an input x, that could either be a single Tensor
            or a nested dict, list of tensors and some additional inputs
        xs: the inputs that're to be mapped over. We'll iterate over the first dim of each x
            and perform f on each slice.

        *args: additional arguments provided to each step of f. They could also be omitted and
            map is able to automatically figure out the read dependency.

    Return:
        the stacked output for each step of f

    Example:

        def f(xs):
            return xs[0] + xs[1] + const1 + const2

        xs = [torch.randn(2, 3), torch.randn(2, 3)]
        const1 = torch.randn(2, 3)
        const2 = torch.randn(2, 3)
        # returns a tensor of shape [2, 2, 3]
        torch._higher_order_ops.map(f, xs)

    """

class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fw_graph, joint_graph, num_mapped_args, *flat_args): ...
    @staticmethod
    def backward(ctx, *flat_grads): ...

def trace_map(proxy_mode, func_overload, f, xs, pos_args): ...
def map_dense(f, xs, pos_args): ...
@map_impl.py_autograd_impl
def map_autograd(f, xs, pos_args): ...
def map_proxy_torch_dispatch_mode(mode, f, xs, args): ...
def map_fake_tensor_mode(mode, f, xs, args): ...
@map_impl.py_functionalize_impl
def map_functionalize(ctx, f, xs, pos_args): ...
def _fake_map(f, x, *args): ...
