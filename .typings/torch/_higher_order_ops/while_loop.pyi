import torch
from _typeshed import Incomplete
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.utils import _maybe_run_with_interpreter as _maybe_run_with_interpreter, _set_compilation_env as _set_compilation_env, autograd_not_implemented as autograd_not_implemented, check_meta_consistency as check_meta_consistency, reenter_make_fx as reenter_make_fx, validate_subgraph_args_types as validate_subgraph_args_types
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, _temp_remove_metadata_torch_function_mode as _temp_remove_metadata_torch_function_mode, track_tensor_tree as track_tensor_tree
from typing import Callable

class WhileLoopOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, cond_fn: Callable, body_fn: Callable, carried_inputs: tuple[torch.Tensor | int | float | bool], additional_inputs: tuple[torch.Tensor | torch.SymInt | int, ...], /): ...

while_loop_op: Incomplete

def while_loop(cond_fn, body_fn, carried_inputs):
    """
    Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or
    initial carried_inputs.

    .. warning::
        `torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `while_loop` is a structured control flow operator. It preserves the loop semantic across the torch.compile and torch.export.

    `while_loop` is equivalent to the following:

        def while_loop(cond_fn, body_fn, carried_inputs):
            val = carried_inputs
            while cond_fn(*val):
                val = body_fn(*val)
            return val

    Args:
        cond_fn (Callable): A callable function that returns a boolean Scalar tensor or a python boolean.

        body_fn (Callable): A callable function that takes the same inputs as `cond_fn` and returns a tuple of tensors or ints

        carried_inputs (Tuple of possibly nested dict/list/tuple of tensors or ints): A tuple of inputs to cond_fn and body_fn.
            It's also the initial value of states that are carried across iterations. Note that when pass an integer as carry,
            the corresponding return of while_loop will be another int with unknown values because we don't know how many
            iterations while_loop will run.

    Example 1:

        def cond_fn(iter, x):
            return iter.sum() < 10

        def body_fn(iter, x):
            return iter + 1, x.sin()

        while_loop(cond_fn, body_fn, (torch.zeros(1), torch.randn(3, 4)))

    Example 2:

        def cond_fn(int_iter, x):
            return 2 * int_iter < x.shape[0]

        def body_fn(int_iter, x):
            return int_iter + 1, x + int_iter

        while_loop(cond,_fn, body_fn, (0, torch.randn(3, 4)))

    Restrictions:

        - body_fn must return tensors or int with the same metadata (e.g.shape, dtype) as inputs.

        - body_fn and cond_fn must not in-place mutate the carried_inputs. A clone before the mutation is required.

        - body_fn and cond_fn must not mutate python varialbles (e.g. list/dict) created outside of the body_fn.

        - body_fn and cond_fn's output cannot aliase any of the inputs. A clone is required.

    .. warning::
        Temporal Limitations:

        - 'while_loop' only supports **inference** right now. Autograd will be supported in the future.

    """
def while_loop_dense(cond_fn, body_fn, carried_inputs, additional_inputs): ...
def _find_or_create_fake_mode() -> FakeTensorMode: ...
def _create_unbacked_symint(fake_mode: FakeTensorMode, ignore_fresh_unbacked_symbols: bool) -> torch.SymInt: ...
def while_loop_tracing(mode, cond_fn, body_fn, carried_inputs, additional_inputs): ...
def while_loop_fake_tensor_mode(mode, cond_fn, body_fn, carried_inputs, additional_inputs): ...
@while_loop_op.py_functionalize_impl
def while_loop_func(ctx, cond_fn, body_fn, carried_inputs, additional_inputs): ...
