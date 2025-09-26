import torch
from torch.fx._symbolic_trace import symbolic_trace as symbolic_trace
from torch.fx.node import Node as Node
from torch.fx.passes.tools_common import legalize_graph as legalize_graph

def split_result_tensors(result: torch.Tensor, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """
    A free function for use in the merge_matmul graph transformation below that
    splits the output from a merged matmul into the individual results for each
    input tensor.

    Arguments:
        result: The merged matmul result tensor.
        inputs: The list of inputs that were merged into one for the matmul.

    Returns:
        List of matmul results for each input tensor.
    """
def may_depend_on(a: Node, b: Node, search_depth: int = 6):
    """
    Determine if one node depends on another in a torch.fx.Graph.

    Arguments:
        a: The node that may have a dependency on b.
        b: The node that a may have a dependency on.
        search_depth: In the case of an indirect dependency, this function
                        searches upto this many nodes away in search of a
                        data dependency. If none is found, the function
                        makes the conservative assumption that there is a
                        dependency.

    Returns:
        True if a may depend on b, False if it definitely does not.
    """
def are_nodes_independent(nodes: list[Node]):
    """
    Check if all of the given nodes are pairwise-data independent.

    Arguments:
        nodes: The nodes to check for data dependencies.

    Returns:
        True if any pair in nodes has a data dependency.
    """
def merge_matmul(in_mod: torch.nn.Module):
    """
    A graph transformation that merges matrix multiplication operations that share the same right-hand
    side operand into one large matrix multiplication.
               ____      _________        _________
      ----    |    |    |         |     M|  A * C  |
    M| A  |  T| B  | * K|    C    | =    |---------|
      ---- ,  |    |    |         |     T|  B * C  |
       K       ----      ---------        ---------
                K            R                R
    """
