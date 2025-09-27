import torch
from ._debug import map_debug_info as map_debug_info
from _typeshed import Incomplete
from collections.abc import Iterator
from torch.autograd.graph import GradientEdge as GradientEdge, Node as Node
from torch.nn import Parameter as Parameter
from typing import Any

logger: Incomplete

def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Node | None:
    """
    Get the grad function or grad accumulator for a tensor.

    Accumulate grad nodes are lazily created, so we need to a
    dummy view in order to trigger its creation.
    """
def reverse_closure(roots: list[Node], target_nodes: set[Node], reverse_edges_dict) -> tuple[set[Node], set[Node]]:
    """
    This function returns the reverse closure of the given roots,
    i.e. the set of nodes that can be reached from the roots by following the
    reverse edges of the graph. The target_nodes are the nodes that we want to
    include in the closure.
    """
def construct_reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]: ...
def get_param_groups(inputs: list[Node], params: list[Node], reverse_edges_dict) -> list[dict[str, Any]]:
    '''
    Given a list of inputs and a list of parameters, return a list of parameter
    groups, where each group contains the parameters and the intermediates that
    are connected to the parameters.

    The returned list of parameter groups is a list of dictionaries, where each
    dictionary contains the following keys:
    - "params": a set of parameters
    - "intermediates": a set of intermediates

    The returned list of parameter groups is a list of dictionaries,
    '''
def stage_backward_input(stage_outputs_or_loss: list[torch.Tensor], output_grads: list[torch.Tensor] | None, input_values: list[torch.Tensor], weights: Iterator[Parameter]) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]]]:
    """
    Compute the gradients for only the stage inputs with
    respect to the stage outputs (if non-last stage) or loss (if last stage)

    After computing input gradients, we save the intermediate nodes in `param_groups`
    for later use in stage_backward_weight. We don't need to save any other intermediate nodes
    that aren't needed for dW because when we do dW calculation, we start from saved intermediates.
    Detaching the stage_outputs_or_loss at the end of this function is important as
    it frees up the memory that the autograd graph is anticipating to be used later (but doesn't actually need).
    """
def stage_backward_weight(weights: Iterator[Parameter], param_groups: list[dict[str, Any]], retain_graph: bool = False) -> tuple[torch.Tensor | None, ...]: ...
def stage_backward(stage_output, output_grads, input_values, outputs_with_grads_idxs: list[int] | None = None) -> tuple[torch.Tensor | None, ...]:
    """
    This is a helper function to:
    1. compute the gradients for the stage inputs, and
    2. accumulate gradients for the stage module's parameters.

    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """
def _null_coalesce_accumulate(lhs, rhs):
    """
    Coalesce two values, even if one of them is null, returning the non-null
    value.
    """
