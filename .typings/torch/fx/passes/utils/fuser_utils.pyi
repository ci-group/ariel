from torch.fx._compatibility import compatibility as compatibility
from torch.fx.graph import Graph as Graph
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.node import Node as Node
from torch.fx.passes.tools_common import NodeList as NodeList, NodeSet as NodeSet, legalize_graph as legalize_graph
from torch.fx.passes.utils import lift_subgraph_as_module as lift_subgraph_as_module

def topo_sort(nodes: NodeList) -> NodeList: ...
def validate_partition(partition: NodeList) -> bool: ...
def fuse_as_graphmodule(gm: GraphModule, nodes: NodeList, module_name: str, partition_lookup_table: dict[Node, None] | None = None, *, always_return_tuple: bool = False) -> tuple[GraphModule, tuple[Node, ...], tuple[Node, ...]]:
    """
    Fuse nodes in graph_module into a GraphModule.

    Args:
        gm (GraphModule): target graph_module

        nodes (List[Node]): list of nodes in `gm` to fuse, where the node must be topologically sorted

        module_name: class name for the fused GraphModule

        partition_lookup_table (Optional[Dict[Node, None]]): optional dict of nodes to speed up lookup

        always_return_tuple (bool): whether to always return a tuple, even if there is only one output

    Returns:
        fused_gm (GraphModule): fused graph module, where its node is a copy of `nodes` in `gm`

        original_inputs (Tuple[Node, ...]): input nodes to `nodes` in original `gm`

        original_outputs (Tuple[Node, ...]): consumer nodes of `nodes` in original `gm`

    """
def insert_subgm(gm: GraphModule, sub_gm: GraphModule, orig_inputs: tuple[Node, ...], orig_outputs: tuple[Node, ...]) -> GraphModule: ...
def erase_nodes(gm: GraphModule, nodes: NodeList) -> None: ...
def fuse_by_partitions(gm: GraphModule, partitions: list[dict[Node, None]], prefix: str = 'fused_', always_return_tuple: bool = False) -> GraphModule: ...
