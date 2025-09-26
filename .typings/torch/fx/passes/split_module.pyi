import torch
from _typeshed import Incomplete
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from typing import Any, Callable

__all__ = ['Partition', 'split_module']

class Partition:
    name: str
    submod_name: Incomplete
    node_names: list[str]
    inputs: dict[str, None]
    outputs: dict[str, None]
    dependencies: dict[str, None]
    dependents: dict[str, None]
    graph: torch.fx.graph.Graph
    environment: dict[Node, Node]
    targets: dict[str, Any]
    def __init__(self, name: str) -> None: ...
    def __repr__(self) -> str: ...

def split_module(m: GraphModule, root_m: torch.nn.Module, split_callback: Callable[[Node], int], qualname_map: dict[str, str] | None = None, keep_original_order: bool | None = False, keep_original_node_name: bool | None = False, keep_original_input_name: bool = True):
    """
    Creates subgraphs out of main graph

    Args:
        m (GraphModule): Graph module to split
        root_m (torch.nn.Module): root nn module. Not currently used. Included
            because the root nn module is usually transformed via
            torch.fx._symbolic_trace.symbolic_trace (see example below)
        split_callback (Callable[[Node], int]): Callable function
            that maps a given Node instance to a numeric partition identifier.
            split_module will use this function as the policy for which operations
            appear in which partitions in the output Module.
        qualname_map: Optional[Dict[str, str]]: optional output parameter that returns a
            mapping from new target names in the module after split to old target
            names in the original module.
        keep_original_order: Optional[bool]: keep the original order of the GraphModule
            or use the Topological order of the new constructed GraphModule
        keep_original_node_name: Optional[bool]: If the partitioned graphs should
            have the same node names as the original graph.
        keep_original_input_name: bool: If the partitioned graphs should
            have the same input names as the original graph.

    Returns:
        GraphModule: the module after split.

    Example:

        This is a sample setup:

            import torch
            from torch.fx.symbolic_trace import symbolic_trace
            from torch.fx.graph_module import GraphModule
            from torch.fx.node import Node
            from torch.fx.passes.split_module import split_module

            class MyModule(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand(3, 4))
                    self.linear = torch.nn.Linear(4, 5)

                def forward(self, x, y):
                    z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
                    w = self.linear(y).clamp(min=0.0, max=1.0)
                    return z + w

            # symbolically trace model
            my_module = MyModule()
            my_module_traced = symbolic_trace(my_module)

            # random mod partitioning
            partition_counter = 0
            NPARTITIONS = 3

            def mod_partition(node: Node):
                global partition_counter
                partition = partition_counter % NPARTITIONS
                partition_counter = (partition_counter + 1) % NPARTITIONS
                return partition

            # split module in module with submodules
            module_with_submodules = split_module(
                my_module_traced, my_module, mod_partition
            )

        Output looks like this. Original graph is broken into partitions

            > print(module_with_submodules)
            GraphModule(
                (submod_0): GraphModule(
                    (linear): Linear(in_features=4, out_features=5, bias=True)
                )
                (submod_1): GraphModule(
                    (linear): Linear(in_features=4, out_features=5, bias=True)
                )
                (submod_2): GraphModule()
            )

            def forward(self, x, y):
                param = self.param
                submod_0 = self.submod_0(x, param, y);  x = param = y = None
                getitem = submod_0[0]
                getitem_1 = submod_0[1];  submod_0 = None
                submod_1 = self.submod_1(getitem, getitem_1);  getitem = getitem_1 = None
                getitem_2 = submod_1[0]
                getitem_3 = submod_1[1];  submod_1 = None
                submod_2 = self.submod_2(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
                return submod_2

        Output of split module is the same as output of input traced module.
        This is an example within a test setting:

            > orig_out = my_module_traced(x, y)
            > submodules_out = module_with_submodules(x, y)
            > self.assertEqual(orig_out, submodules_out)
            True
    """
