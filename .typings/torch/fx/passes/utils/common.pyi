from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.nn import Module

__all__ = ['HolderModule', 'lift_subgraph_as_module', 'compare_graphs']

class HolderModule(Module):
    """
    HolderModule is used to copy all the attributes from original module to submodules
    that uses the attributes
    """
    def __init__(self, d) -> None: ...

def lift_subgraph_as_module(gm: GraphModule, subgraph: Graph, comp_name: str = '', class_name: str = 'GraphModule') -> tuple[GraphModule, dict[str, str]]:
    """
    Create a GraphModule for subgraph, which copies the necessary attributes from the original parent graph_module.

    Args:
        gm (GraphModule): parent graph module

        subgraph (Graph): a valid subgraph that contains copied nodes from the parent graph

        comp_name (str): name for the new component

        class_name (str): name for the submodule

    """
def compare_graphs(left: Graph, right: Graph) -> bool:
    """
    Return True if two graphs are identical, i.e they
        - have the same number of outputs in the same order
        - have the same number of inputs in the same order
        - have the same set of nodes, and identical connectivity
    """
