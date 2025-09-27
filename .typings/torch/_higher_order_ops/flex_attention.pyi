import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import Tensor as Tensor
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.utils import UnsupportedAliasMutationException as UnsupportedAliasMutationException, _has_potential_branch_input_mutation as _has_potential_branch_input_mutation, _maybe_reenter_make_fx as _maybe_reenter_make_fx, autograd_not_implemented as autograd_not_implemented, has_user_subclass as has_user_subclass, redirect_to_mode as redirect_to_mode, reenter_make_fx as reenter_make_fx, register_fake as register_fake, save_tensors_and_symints_for_backward as save_tensors_and_symints_for_backward, saved_tensors_and_symints as saved_tensors_and_symints, validate_subgraph_args_types as validate_subgraph_args_types
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses import FakeTensor as FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor as FunctionalTensor
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, make_fx as make_fx, track_tensor_tree as track_tensor_tree
from torch.fx.graph_module import GraphModule as GraphModule
from torch.utils.checkpoint import _CachedTorchDispatchMode as _CachedTorchDispatchMode, _CachingTorchDispatchMode as _CachingTorchDispatchMode
from typing import Any, Callable

def _construct_strides(sizes: Sequence[int], fill_order: Sequence[int]) -> Sequence[int]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
def _permute_strides(out: torch.Tensor, query_strides: tuple[int, ...]) -> torch.Tensor:
    """
    Create a new tensor with the same data and shape as the input,
    but with strides permuted based on the input tensor's stride order.

    Args:
        out (torch.Tensor): The output tensor of attention.
        query_strides (List[int]): The stride order of the input query tensor

    Returns:
        torch.Tensor: A new tensor with same shape and data as the input,
        but with strides permuted based on the query tensor's stride order.
    """

class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]: ...

flex_attention: Incomplete

class FlexAttentionBackwardHOP(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, out: torch.Tensor, logsumexp: torch.Tensor, grad_out: torch.Tensor, grad_logsumexp: torch.Tensor, fw_graph: Callable | GraphModule, joint_graph: GraphModule, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]: ...

flex_attention_backward: Incomplete

def _math_attention_inner(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]: ...
def math_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager implementation

    This implementation uses vmap to vectorize the score_mod function over the batch, head, m, and n dimensions.
    We then apply the vectorized score_mod function to the scores matrix. Each wrap of vmap applies one of the
    batch, head, m, or n dimensions. We need to apply vmap 4 times to vectorized over all 4 dimensions.

    Args:
        query: The query tensor
        key: The key tensor
        value: The value tensor
        score_mod: The score_mod function
        other_buffers: Other buffers that are passed to the score_mod function
    """
def sdpa_dense(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]: ...
def trace_flex_attention(proxy_mode: ProxyTorchDispatchMode, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]:
    '''Traces the flex_attention operator with the given score_mod function and other_buffers.

    Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
    This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
    access this graph module in inductor to inline the score_mod function to the triton template.
    '''
def flex_attention_proxy_torch_dispatch_mode(mode: ProxyTorchDispatchMode, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]: ...
@flex_attention.py_functionalize_impl
def flex_attention_functionalize(ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the flex_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
def flex_attention_fake_impl(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]: ...
def create_fw_bw_graph(score_mod: Callable, index_values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], other_buffers: tuple[Tensor, ...]) -> tuple[Callable, Callable]: ...

class FlexAttentionAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, query: Tensor, key: Tensor, value: Tensor, fw_graph: Callable, joint_graph: Callable, block_mask: tuple[Any, ...], scale: float, kernel_options: dict[str, Any], mask_mod_other_buffers: tuple[Any, ...], *score_mod_other_buffers: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def backward(ctx: Any, grad_out: Tensor, grad_logsumexp: Tensor) -> tuple[Tensor | None, ...]: ...

def flex_attention_autograd(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple[Tensor, ...] = (), mask_mod_other_buffers: tuple[Tensor, ...] = ()) -> tuple[torch.Tensor, torch.Tensor]: ...
def sdpa_dense_backward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, out: torch.Tensor, logsumexp: torch.Tensor, grad_out: torch.Tensor, grad_logsumexp: torch.Tensor, fw_graph: Callable, joint_graph: Callable, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple, mask_mod_other_buffers: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]: ...
def trace_flex_attention_backward(proxy_mode: ProxyTorchDispatchMode, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, out: torch.Tensor, logsumexp: torch.Tensor, grad_out: torch.Tensor, grad_logsumexp: torch.Tensor, fw_graph: Callable | GraphModule, joint_graph: GraphModule, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """We already have the forward graph and joint graph from the forward pass, so we create a proxy attach both graphs"""
def flex_attention_backward_proxy_torch_dispatch_mode(mode: ProxyTorchDispatchMode, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, out: torch.Tensor, logsumexp: torch.Tensor, grad_out: torch.Tensor, grad_logsumexp: torch.Tensor, fw_graph: Callable | GraphModule, joint_graph: GraphModule, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]: ...
@flex_attention_backward.py_functionalize_impl
def flex_attention_backward_functionalize(ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, out: torch.Tensor, logsumexp: torch.Tensor, grad_out: torch.Tensor, grad_logsumexp: torch.Tensor, fw_graph: Callable | GraphModule, joint_graph: GraphModule, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Defines the functionalization rules for the flex_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next,
    since we know that the forward score mod function is assured to be free of mutations
    to the other_buffers, we skip that mutate check and go straight to redispatching.
    """
def flex_attention_backward_fake_tensor_mode(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, out: torch.Tensor, logsumexp: torch.Tensor, grad_out: torch.Tensor, grad_logsumexp: torch.Tensor, fw_graph: Callable | GraphModule, joint_graph: GraphModule, block_mask: tuple, scale: float, kernel_options: dict[str, Any], score_mod_other_buffers: tuple = (), mask_mod_other_buffers: tuple = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]: ...
