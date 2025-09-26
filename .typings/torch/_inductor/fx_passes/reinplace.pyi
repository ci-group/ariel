import torch
import torch.fx
from _typeshed import Incomplete
from dataclasses import dataclass
from torch._C._dynamo.guards import compute_overlapping_tensors as compute_overlapping_tensors
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._dynamo.utils import ReInplaceTrigger as ReInplaceTrigger, ReinplaceCounters as ReinplaceCounters
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table as kernel_side_table, triton_kernel_wrapper_functional as triton_kernel_wrapper_functional
from torch._inductor import config as config, inductor_prims as inductor_prims
from torch._inductor.fx_utils import get_node_storage as get_node_storage, is_node_realized as is_node_realized
from torch._inductor.virtualized import V as V
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode as GuardOnDataDependentSymNode
from torch.fx.immutable_collections import immutable_dict as immutable_dict, immutable_list as immutable_list
from torch.fx.passes.reinplace import _is_view_op as _is_view_op
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable

log: Incomplete
aten: Incomplete

@dataclass(frozen=True)
class InplaceableOp:
    inplace_op: Callable[..., Any]
    mutated_arg: int
    extra_check: Callable[[torch.fx.Node], bool] = ...

_SCATTER_OP_TO_VIEW: Incomplete
_VIEW_OP_TO_SCATTER: Incomplete

def graph_call_function(graph: torch.fx.Graph, fn, *args, **kwargs): ...

@dataclass
class ViewOp:
    target: torch._ops.OpOverload
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

def _inplace_generalized_scatter(inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]) -> torch.Tensor: ...
def _generalized_scatter(inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]) -> torch.Tensor: ...
def _decompose_scatter_functional_helper(graph: torch.fx.Graph, inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]) -> torch.fx.Node: ...
def _decompose_scatter_functional(graph: torch.fx.Graph, node: torch.fx.Node) -> torch.fx.Node:
    """Decompose _generalized_scatter to a sequence of view_scatter operations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    view = aten.slice(inp, 0, 0, 10)
    view_updated = aten.slice_scatter(view, src, 1, 10, -10)
    inp_updated = aten.slice_scatter(inp, view_updated, 0, 0, 10)
    """
def _decompose_scatter_mutating(graph: torch.fx.Graph, node: torch.fx.Node) -> torch.fx.Node:
    """Decompose _generalized_scatter using mutations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    inp_updated = aten.clone(inp)
    slice1 = aten.slice(inp_updated, 0, 0, 10)
    slice2 = aten.slice(slice1, 1, 10, -10)
    slice2.copy_(src)

    """

_ALWAYS_MUTATING_SCATTER_OPS: Incomplete

def scatter_always_uses_mutation(node: torch.fx.Node) -> bool: ...
def should_reinplace_scatter(node: torch.fx.Node) -> bool:
    """Choose between mutating and functional scatter decompositions

    Reinplacing view scatter ops can be pessimising as it blocks fusion with the
    input or output tensor computations. However, it is still profitable if the
    input and output would have been realized anyway.

    """
def decompose_generalized_scatter(graph: torch.fx.Graph) -> None:
    """Replace _generalized_scatter with normal aten ops"""
def canonicalize_view_scatter_ops(graph: torch.fx.Graph) -> None:
    """
    This canonicalizes view scatter ops into a generalized form, defined as:
      def scatter(inp, src, views):
        tmp = inp.clone()
        for view in views:
          tmp = view(tmp)
        tmp.copy_(src)

    We also fuse consecutive view scatter ops of the form
        a = scatter(view2(self), src, [view1])
        b = scatter(self, a, [view2])
    which can be rewritten as
        b = scatter(self, src, [view2, view1])
        a = view2(b)

    This is both more efficient as we only do a single scatter, and also
    easier to reinplace since there is only one use of `self`
    """

inplaceable_ops: dict[Callable[..., Any], InplaceableOp]
c10d_functional: Incomplete
inplaceable_collective_ops: dict[Callable[..., Any], InplaceableOp]
inplaceable_foreach_ops: dict[torch._ops.OpOverload, InplaceableOp]
inplaceable_triton_ops: Incomplete
META_ONLY_OPS: Incomplete

def reinplace_inplaceable_ops_core(graph: torch.fx.Graph) -> None:
    """
    Reinplaces in-placeable operations.
    If there are no uses of a view of the mutated arg after the current node,
    it is possible to inplace the op.
    This above algorithm could be justified by observing side effects. While
    we traverse the graph in forwards direction, only latter nodes could view
    side effects of the current node. If the current node is not used later as
    well as no view of this node is used later in the graph, then it is safe to
    inplace as there would be no way to observe the side effects.
    This condition is slightly different for graph inputs where they can only
    be inplaced if the above condition is true and there's a copy_ in the
    epilogue that signals that the caller wants to observe the mutation.

    Unlike JIT Inductor, AOTInductor currently unlifts weights and buffers from
    input args, so instead of checking mutation on placeholder, AOTInductor
    checks mutation on get_attr. This is subject to change in future.
    """
def reinplace_inplaceable_ops(graph: torch.fx.Graph) -> None: ...
