import torch
from .utils import clone_outputs_aliasing_inputs as clone_outputs_aliasing_inputs
from _typeshed import Incomplete
from torch._C import DispatchKey as DispatchKey
from torch._C._functorch import _add_batch_dim as _add_batch_dim, get_unwrapped as get_unwrapped, is_batchedtensor as is_batchedtensor, maybe_get_bdim as maybe_get_bdim
from torch._functorch.utils import exposed_in as exposed_in
from torch._higher_order_ops.utils import _maybe_run_with_interpreter as _maybe_run_with_interpreter, _set_compilation_env as _set_compilation_env, materialize_as_graph as materialize_as_graph, reenter_make_fx as reenter_make_fx, save_tensors_and_symints_for_backward as save_tensors_and_symints_for_backward, saved_tensors_and_symints as saved_tensors_and_symints, unique_graph_id as unique_graph_id, validate_subgraph_args_types as validate_subgraph_args_types
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor, FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, _temp_remove_metadata_torch_function_mode as _temp_remove_metadata_torch_function_mode, _temp_remove_pre_dispatch_torch_function_mode as _temp_remove_pre_dispatch_torch_function_mode, track_tensor_tree as track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode as _get_current_dispatch_mode
from typing import Any, Callable

log: Incomplete

class CondOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, pred, true_fn, false_fn, operands): ...

cond_op: Incomplete

def cond(pred: bool | int | float | torch.Tensor, true_fn: Callable, false_fn: Callable, operands: tuple | list = ()) -> Any:
    """
    Conditionally applies `true_fn` or `false_fn`.

    .. warning::
        `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `cond` is structured control flow operator. That is, it is like a Python if-statement,
    but has restrictions on `true_fn`, `false_fn`, and `operands` that enable it to be
    capturable using torch.compile and torch.export.

    Assuming the constraints on `cond`'s arguments are met, `cond` is equivalent to the following::

        def cond(pred, true_branch, false_branch, operands):
            if pred:
                return true_branch(*operands)
            else:
                return false_branch(*operands)

    Args:
        pred (Union[bool, torch.Tensor]): A boolean expression or a tensor with one element,
          indicating which branch function to apply.

        true_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced.

        false_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced. The true branch and false branch must
          have consistent input and outputs, meaning the inputs have to be
          the same, and the outputs have to be the same type and shape. Int
          output is also allowed. We'll make the output dynamic by turning it
          into a symint.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): A tuple of inputs to the
          true/false functions. It can be empty if true_fn/false_fn doesn't require input. Defaults to ().

    Example::

        def true_fn(x: torch.Tensor):
            return x.cos()
        def false_fn(x: torch.Tensor):
            return x.sin()
        return cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    Restrictions:
        - The conditional statement (aka `pred`) must meet one of the following constraints:

          - It's a `torch.Tensor` with only one element, and torch.bool dtype

          - It's a boolean expression, e.g. `x.shape[0] > 10` or `x.dim() > 1 and x.shape[1] > 10`

        - The branch function (aka `true_fn`/`false_fn`) must meet all of the following constraints:

          - The function signature must match with operands.

          - The function must return a tensor with the same metadata, e.g. shape,
            dtype, etc.

          - The function cannot have in-place mutations on inputs or global variables.
            (Note: in-place tensor operations such as `add_` for intermediate results
            are allowed in a branch)

    """
def create_bw_fn(fn: Callable, args: tuple[Any]) -> Callable:
    """
    For a fn that accepts flat inputs and returns flat outputs:
        fw_out = fn(*args),
    this function returns:
        grad_args = bw_fn(*args_and_grad_output)
    with the following invariants:
      1. args + fw_out has an 1-1 correspondence to args_and_grad_output
      2. grad_args has an 1-1 corresponsence to args
      3. for tensor arg whose requires_grad is False, its corresponding grad in
         grad_args will be a zero tensor with the same shape.
    """
def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands): ...
def cond_op_dense(pred, true_fn, false_fn, operands): ...

class CondAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, true_fn, false_fn, *operands): ...
    @staticmethod
    def backward(ctx, *flat_grads): ...

@cond_op.py_autograd_impl
def cond_autograd(pred, true_fn, false_fn, operands): ...
def inner(mode, pred, true_fn, false_fn, operands): ...
def cond_fake_tensor_mode(mode, pred, true_fn, false_fn, operands): ...
def check_tensor_meta_match(t1: torch.Tensor, t2: torch.Tensor, attr_names: tuple[str, ...], msg_prefix: str) -> None: ...
def _merge_output(a: torch.Tensor | int | None, b: torch.Tensor | int | None, mode: FakeTensorMode): ...
@cond_op.py_functionalize_impl
def cond_func(ctx, pred, true_fn, false_fn, inputs): ...
def cond_batch_rule(interpreter, pred, true_fn, false_fn, inputs): ...
