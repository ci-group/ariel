import torch
import torch.fx
from _typeshed import Incomplete
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ['get_acc_ops_name', 'get_node_target', 'is_node_output_tensor', 'FxNetAccFusionsFinder', 'legalize_graph']

Tensors = tuple[torch.Tensor] | list[torch.Tensor]
TensorOrTensors = torch.Tensor | Tensors
NodeList = list[torch.fx.Node]
NodeSet = set[torch.fx.Node]
Names = list[str]

def get_acc_ops_name(k): ...
def get_node_target(submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> str:
    '''
    Given a `node` returns its target typename.

    For "call_method" node, return node.target which is the name of that method being called.
    This could potential lead to conflict but should be okay because normally it\'s on a tensor.

    For "call_function" node, return typename of node.target.

    For "call_module" node, return typename of the module that node.target point to.

    If seeing "_VariableFunctionsClass" in the target name string, it will be replaced by
    "torch". e.g. _VariableFunctionsClass.relu would become torch.relu.
    '''
def is_node_output_tensor(node: torch.fx.Node) -> bool:
    """Checks if the node output produces a Tensor or not.

    NOTE: This requires to run `ShapeProp` on the containing fx graph before
    calling this function. This is because it works by checking the `type`
    metadata on the node. This metadata is produced by the `ShapeProp`.
    """

class FxNetAccFusionsFinder:
    """
    Finds groups of connected ACC nodes that pass non-tensor data between each other.
    Such groups are called fusion groups.
    """
    module: Incomplete
    nodes: Incomplete
    acc_nodes: Incomplete
    def __init__(self, module: torch.fx.GraphModule, acc_nodes: NodeSet) -> None: ...
    @dataclass
    class FusionGroup:
        top_node_idx: int
        nodes: NodeSet
        inputs: NodeSet
        nodes_need_process: NodeSet
        def add_node(self, node) -> None:
            """
            Add a node to fusion group.
            """
    def recursive_add_node(self, fusion_group: FxNetAccFusionsFinder.FusionGroup, inputs: NodeSet | NodeList, visited: NodeSet | None = None):
        """
        Start from inputs and going reverse topological order. If any upstream node
        is in the fusion group, add all the nodes in this path to fusion group.
        """
    def __call__(self) -> dict[torch.fx.Node, NodeSet]: ...

def legalize_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Replace the graph of the given GraphModule with one that contains the same nodes as the
    original, but in topologically sorted order.

    This is used by the merge_matmul transformation below, which disturbs the topologically sorted
    order of its input GraphModule, so that this order is restored before further transformation.

    Arguments:
        gm: The graph module to topologically sort. It is modified in-place.

    Returns:
        The graph module in-place sorted
    """
