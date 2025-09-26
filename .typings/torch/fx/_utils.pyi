import torch
from torch._logging import LazyString as LazyString

def lazy_format_graph_code(name, gm, maybe_id=None, **kwargs):
    """
    Returns a LazyString that formats the graph code.
    """
def _format_graph_code(name, filename, graph_str):
    """
    Returns a string that formats the graph code.
    """
def first_call_function_nn_module_stack(graph: torch.fx.Graph) -> dict | None:
    """
    Returns the nn_module_stack of the first call_function node.
    """
def get_node_context(node, num_nodes: int = 2) -> str:
    """
    Returns a string of the last num_nodes nodes in the graph.
    """
