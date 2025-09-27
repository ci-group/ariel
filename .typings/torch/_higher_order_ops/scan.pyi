import torch
import torch.utils._pytree as pytree
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.cond import create_bw_fn as create_bw_fn
from torch._higher_order_ops.utils import _maybe_compile_and_run_fn as _maybe_compile_and_run_fn, check_meta_consistency as check_meta_consistency, first_slice_copy as first_slice_copy, materialize_as_graph as materialize_as_graph, reenter_make_fx as reenter_make_fx, save_tensors_and_symints_for_backward as save_tensors_and_symints_for_backward, saved_tensors_and_symints as saved_tensors_and_symints, unique_graph_id as unique_graph_id, validate_subgraph_args_types as validate_subgraph_args_types
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, track_tensor_tree as track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode as _get_current_dispatch_mode
from typing import Any, Callable

aten: Incomplete

def wrap_combine_fn_flat(*args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves): ...
def _extract_carry_and_out(flat_out: list[Any], num_carry: int): ...
def stack_y(y: torch.Tensor, scan_length: int) -> torch.Tensor: ...
def get_tensor_mask(tensor_list: list[Any]) -> list[bool]: ...
def mask_list(mask: list[bool], inp: list[Any], other: list[Any] | None = None) -> list[Any]: ...
def first_slice_copy_with_grad(li: list[Any]) -> list[Any]: ...
def split_into_chunks(iterable: Sequence[Any], chunk_sizes: list[int]) -> list[Any]: ...
def call_operator(operator, *args): ...
def scan(combine_fn: Callable[[pytree.PyTree, pytree.PyTree], tuple[pytree.PyTree, pytree.PyTree]], init: pytree.PyTree, xs: pytree.PyTree, *, dim: int = 0, reverse: bool = False) -> tuple[pytree.PyTree, pytree.PyTree]:
    """
    Performs an inclusive scan with a combine function.

    .. warning::
        `torch.scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> (Tensor, Tensor)``,
            or if xs is a pytree ``(pytree, pytree) -> (pytree, pytree)``.
            The first input to ``combine_fn`` is the previous or initial scan carry
            and the second input element to ``combine_fn`` is a slice of the input along dim.
            The first output element of ``combine_fn`` is the next scan carry
            and the second output  of ``combine_fn`` represents a slice of the output.
            This function must be pure, i.e., no lifted arguments are supported at the moment
            and may not have any side effects.
        init (torch.Tensor or pytree with tensor leaves): The inital scan carry, a tensor, or nested pytree of tensors.
            The ``init`` is expected to have the same pytree structure as the first output element (i.e. carry)
            of ``combine_fn``.
        xs (torch.Tensor or pytree with tensor leaves): The input tensor, or nested pytree of tensors.

    Kwargs:
        dim (int): the dimension to scan over, default 0.
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.

    Returns:
        final_carry (torch.Tensor or pytree with tensor leaves),
            the final carry of the scan operation with same pytree structure as init.
        out (torch.Tensor or pytree with tensor leaves),
            each tensor leaf is a stacked output along first dim, where each slice is the output of a scan iteration.

    Restrictions:
        - The combine_fn shouldn't have any aliasing between input-input, input-output, and output-output. E.g. return a view
            or the same tensor as input is not supported. As a workaround, can clone the output to avoid aliasing.

        - The combine_fn shoudn't mutate any inputs. We'll remove the mutation restriction for inference soon. Please file an issue
            if you input mutation support for training is needed.

        - The combine_fn's init carry should match the next_carry in pytree structure and in tensor metadata.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            # clone the output to avoid output-output aliasing
            return next_carry, y.clone()

        i0 = torch.zeros(1)
        xs = torch.arange(5)
        # returns torch.tensor([10.]), torch.tensor([[0], [1.], [3.], [6.], [10.]])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """

class ScanOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, combine_fn, init, xs, additional_inputs): ...

scan_op: Incomplete

def generic_scan(operator, init, xs, dim: int = 0, additional_inputs=()): ...
def trace_scan(proxy_mode, func_overload, combine_fn: Callable, init: list[torch.Tensor], xs: list[torch.Tensor], additional_inputs: tuple[torch.Tensor]): ...
def scan_op_dense(combine_fn, init, xs, additional_inputs): ...

class ScanAutogradOp(torch.autograd.Function):
    """
    Example ::

        def combine_fn(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x * y
            return next_carry, y

        The ``combine_fn_bw``, computing the gradients for x and y of ``combine_fn`` is computed as:
        def combine_fn_bw(x: torch.Tensor, y: torch.Tensor, g_carry: torch.Tensor, g_y: torch.Tensor):
            return g_y * y + g_carry * y, g_y * x + g_carry * x

        Note: In a real usecase of scan, there may be additional_inputs that participate in the
        forward as well as in the backward of the scan operator. For the sake of readability those inputs
        have been omitted in the following example, but are included in the subsequent detailed description below

        The forward output of scan is computed as:
        carry, ys = scan(combine_fn, init, xs).

        This computation can be unpacked as
        c_0, ys_0 = combine_fn(init, xs_0)
        c_1, ys_1 = combine_fn(carry_0, xs_1)
        c_2, ys_2 = combine_fn(carry_1, xs_2)
        ...
        c_T, ys_T = combine_fn(carry_(T-1), xs_T)

        We collect c_0, c_1, ..., c_T into a vector of carries that we save for the backward,
        but we only output (c_T, ys),
        where ys is the vector of all intermediate outputs [y_0, y_1, ..., y_T].

        Given the carries and the ys, the gradients for xs and for init can be computed as follows:
        We receive the upstream gradients in torch.autograd.Function, i.e., we get g_c_T and g_ys,
        where g_ys is the vector of all intermediate gradients of the outputs [g_ys_0, g_ys_1, ..., g_ys_T]

        We then proceed to compute the gradients for the init (g_init) and the xs (g_xs) by running a
        scan operation reverse over time. For example,

        g_c_(T-1), g_xs_T = combine_fn_bw(c_(T-1), xs_T, g_c_T, g_ys_T)
        g_c_(T-2), g_xs_(T-1) = combine_fn_bw(c_(T-2), xs_(T-1), g_c_(T-1), g_ys_(T-1))
        g_c_(T-3), g_xs_(T-2) = combine_fn_bw(c_(T-3), xs_(T-2), g_c_(T-2), g_ys_(T-2))
        ...
        g_init, g_xs_1 = combine_fn_bw(c_0, xs_1, g_c_0, g_ys_1)
        0     , g_xs_0 = combine_fn_bw(init, xs_0, g_init, g_ys_0),

        where combine_fn_bw takes the forward inputs of step t (i.e. c_(t-1), xs_t),
        the gradients of the carry of step t (i.e. g_c_t) and
        the upstream gradient of the output of step t (i.e. g_ys_T)
        and returns the gradient of xs_t -> g_xs_t, as well as the gradient for the carry of step t-1 -> g_c_(t-1).

        Through this procedure we end up with the
        gradients for the init -> g_init,
        the gradients for the xs -> g_xs.


    NOTE: [scan autograd implementation]

    The forward of scan can be computed as:
    1.) Prepare the forward graph wrapper ``combine_fn_with_carry_checkpoint``:
    To use a scan operation for the backward path as well, we need access to the carries from all steps.
    Thus, the function ``combine_fn`` is wrapped such that it returns all carries and not only the last carry.
    In particular, we define ``combine_fn_with_carry_checkpoint``:
    def combine_fn_with_carry_checkpoint(x: torch.Tensor, y: torch.Tensor):
        carry, y = combine_fn(x, y)
        return carry, (carry, y)

    The scan operator will stack all outputs along the scan dimension.
    Thus, by putting next_carry also into outputs of ``combine_fn_with_carry_checkpoint``,
    the carries from all steps will be stacked and hence gives us chekpointed_carries

    2.) Compute all carries, the last carry and all outputs using ``combine_fn_with_carry_checkpoint``:
    c_T, (carries, ys) = scan_op(combine_fn_with_carry_checkpoint, init, xs, additional_inputs),
    Where c_T (last carry) and ys (all outputs) are the original results of scan with the ``combine_fn``.
    However, carries are checkpointed carries from all steps.
    As a result of the forward, only the last carry c_T and the ys are returned,
    while all carries are saved for the backward.

    The backward of scan can be computed as:

    3.) Prepare the backward graph:
    We prepare the backward graph to be used in the backward function.
    We utilize ``create_bw_fn`` to generate the joint function, i.e.,
    ctx._combine_fn_bw = create_bw_fn(ctx._combine_fn, fw_operands), where fw_operands = [init, xs_0, additional_inputs]

    The ctx._combine_fn_bw requires the primals (operands)
    followed by the tangents (upstream gradients) from a single step
    and produces the gradients of that step, i.e.,
    g_c_(T-1), g_xs_T, g_additional_input_T = ctx._combine_fn_bw(c_(T-1), xs_T, additional_inputs, g_c_T, g_ys_T).

    4.) Create a wrapper of the ``combine_fn_bw``, i.e., ``combine_fn_bw_grad_accumulation``:
    In the forward, there may be additional inputs that participate in every forward step.
    The gradients for those additional inputs are also computed at every step and need to be accumulated over all steps,
    which is taken care of in this wrapper. For example:
    def combine_fn_bw_grad_accumulation(*args):
        carried_g_additional_input = args[:num_additional_inputs]
        inputs_bw_fn = args[num_additional_inputs:]
        g_c_(t-1), g_xs_t, g_additional_input_t = ctx._combine_fn_bw(*inputs_bw_fn)
        new_g_additional_inputs = carried_g_additional_input + g_additional_input_t
        # The ``new_g_additional_inputs`` and the ``g_c_t`` are encoded in the carry of the backward scan operator
        # The ``g_xs_t`` is encoded as the output of the backward scan operator
        return [*new_g_additional_inputs, *g_c_t, *g_xs_t]

    5.) Perform the backward scan as
    g_additional_inputs, g_init, g_xs = scan_op(combine_fn_bw_grad_accumulation, bw_init, bw_xs), where
    bw_init consists of the initial gradient carry for the additional_inputs (initialized with 0s):
    initial_g_additional_inputs, and the gradient of the last carry: g_c_T. Thus:
    bwd_init = [*initial_g_additional_inputs, *g_c_T].

    bw_xs consists of the combination of the upstream gradients g_ys,
    the forward carries prepended with the fw_init, i.e., bw_carries = concat([fw_init, fw_carries[:-1]]) and
    the fw_xs. In particular,
    bwd_xs = [*g_ys, *bw_carries, *fw_xs].

    Note: g_c_T and g_ys are provided through the torch.autograd.Function.backward's input

    As demonstrated in the Example above, this backward scan then yields the gradient for the init -> g_init
    and the gradient for the xs -> g_xs

    NOTE: [scan partial grad handling]
    If any element of init, of xs, of the outputs or of the additional_inputs does not require gradients,
    i.e., requires_grad=False, there will be still gradients returned for those elements,
    but those gradients will be a tensor filled with zeros of the same shape as the element itself.

    A special case are additional_inputs that are not tensors. Such inputs can occur for example with symbolic tracing,
    where the shape symbol (SymInt) becomes an additional_input.
    For such cases, we compute a ``additional_inputs_tensor_mask``, which is True for elements of additional_inputs
    that are tensors and False otherwise. Gradients of additional_inputs are only accumulated if this mask is True,
    otherwise, the value of initial_g_additional_inputs is passed, which is None for non-Tensor values.
    """
    @staticmethod
    def forward(ctx, combine_fn, num_leaves_init, num_leaves_xs, num_additional_inputs, *operands): ...
    @staticmethod
    def backward(ctx, *flat_grads):
        """
        This function computes the gradients of the scan operation.
        It does so by using a scan operator using all carries and the upstream gradients (see description above)

        Args:
            flat_grads (torch.Tensor): The tensor of flattened upstream gradients.
        """

@scan_op.py_autograd_impl
def scan_autograd(combine_fn, init, xs, additional_inputs): ...
def scan_proxy_mode(mode, combine_fn, init, xs, additional_inputs): ...
def scan_fake_tensor_mode(mode, combine_fn, init, xs, additional_inputs): ...
@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, init, xs, additional_inputs): ...
def _fake_scan(combine_fn, init, xs=None, dim: int = 0, reverse: bool = False): ...
