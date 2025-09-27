import torch
import torch.utils._pytree as pytree
from _typeshed import Incomplete
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.utils import _maybe_compile_and_run_fn as _maybe_compile_and_run_fn, _maybe_run_with_interpreter as _maybe_run_with_interpreter, autograd_not_implemented as autograd_not_implemented, check_meta_consistency as check_meta_consistency, first_slice_copy as first_slice_copy, reenter_make_fx as reenter_make_fx, unique_graph_id as unique_graph_id, validate_subgraph_args_types as validate_subgraph_args_types
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, track_tensor_tree as track_tensor_tree
from typing import Callable

aten: Incomplete

def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves): ...
def _interleave(a, b, dim: int = 0): ...
def safe_map(f, *args): ...

class AssociativeScanOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, combine_fn, xs, additional_inputs): ...

associative_scan_op: Incomplete

def associative_scan(combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree], xs: pytree.PyTree, dim: int, reverse: bool = False, combine_mode: str = 'pointwise') -> torch.Tensor:
    """
    Performs an inclusive scan with an associative combine function.

    .. warning::
        `torch.associative_scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    This operator requires runtime code generation and so requires support for
    ``torch.compile``. Further, only CUDA device codegen is supported at the moment.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment,
            satisfy the associative property and have no side-effects.
        xs (torch.Tensor): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``, default ``pointwise``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure, may only contain pointwise operations
            and ``xs`` must be CUDA tensors.
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """
def generic_associative_scan(operator, leaves, dim: int = 0, additional_inputs=()):
    """
    This function performs the associative_scan operation.
    The algorithm works by recursively collecting neighbours of ``leaves`` and subsequently
    applying the ``operator`` on all pairs in parallel along ``dim``.
    The results of the recursive calls are later combined.

    Args:
        operator (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, pointwise, and satisfy the associative property.
        leaves (torch.Tensor): A list of torch.Tensors converted from the pytree of
            ``xs`` provided to ``associative_scan``.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        additional_inputs (Tuple of tensors): A tuple of lifted parameters from the global scope.
            This parameter will be populated internally.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        leaves = torch.tensor([0.0, 1.0, 2.0, 3.0])

        First iteration of _scan ->
            # odd_elems -> apply operator on all neighbours
            # odd_elems = operator([torch.tensor([0.0, 2.0])],
            #                      [torch.tensor([1.0, 3.0])])
            odd_elems = torch.tensor([1.0, 5.0])
            Second iteration of _scan ->
                # odd_elems = operator([torch.tensor([1.0])],
                #                      [torch.tensor([5.0])])
                odd_elems = torch.tensor([6.0])
                # even_elems -> apply operator on all odd_elems and
                # every second element of ``elems``, starting from the second element.
                # even_elems is expanded with the first element of ``elems``
                even_elems = [1.0]
                # Merges odd_elems and even_elems
                res = torch.tensor([1.0, 6.0])
            # even_elems -> apply operator on all odd_elems and
            # every second element of ``elems``, starting from the second element.
            # even_elems is expanded with the first element of ``elems``
            even_elems = [0.0, 3.0]
            # Merges odd_elems and even_elems
            res = torch.tensor([0.0, 1.0, 3.0, 6.0])

    """
def trace_associative_scan(proxy_mode, func_overload, combine_fn: Callable, xs: list[torch.Tensor], additional_inputs: tuple[torch.Tensor]): ...
def associative_scan_op_dense(combine_fn, xs, additional_inputs): ...
def associative_scan_proxy_mode(mode, combine_fn, xs, additional_inputs): ...
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, xs, additional_inputs): ...
@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, xs, additional_inputs): ...
def _fake_associative_scan(combine_fn, xs, dim, reverse: bool = False): ...
