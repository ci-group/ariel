import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, Target
from typing import Any, NamedTuple

__all__ = ['replace_target_nodes_with', 'size_bytes', 'get_size_of_all_nodes', 'get_tensor_meta', 'get_size_of_node']

def replace_target_nodes_with(fx_module: GraphModule, old_op: str, old_target: Target, new_op: str, new_target: Target):
    """Modifies all nodes in fx_module.graph.nodes which match the specified op code and target,
    and updates them to match the new op code and target"""

class size_bytes(NamedTuple):
    output_size: int
    total_size: int

def get_size_of_all_nodes(fx_module: GraphModule, args: list[torch.Tensor] | None = None) -> None:
    """Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size"""
def get_tensor_meta(node: Node) -> Any: ...
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size
    """
