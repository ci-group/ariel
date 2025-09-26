import torch
from ..utils import node_inline_ as node_inline_, nodes_filter as nodes_filter, nodes_first as nodes_first, sequential_split as sequential_split
from .replace_with_hop_pass_util import _replace_with_hop_helper as _replace_with_hop_helper, _replace_with_hop_pass_helper as _replace_with_hop_pass_helper, _sequential_split_and_maybe_inline_subgraphs_helper as _sequential_split_and_maybe_inline_subgraphs_helper
from torch._higher_order_ops.wrap import wrap_with_autocast as wrap_with_autocast
from torch.export.graph_signature import ExportGraphSignature as ExportGraphSignature

def _is_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool: ...
def _is_enter_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool: ...
def _is_exit_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool: ...
def _is_autocast_sub_mod(node: torch.fx.Node) -> bool:
    """
    Check if the first non-placeholder node is `torch.amp.autocast_mode._enter_autocast`.
    """
def _check_valid_autocast_block(enter_autocast_node: torch.fx.Node, exit_autocast_node: torch.fx.Node) -> None: ...
def _replace_with_hop(node: torch.fx.Node) -> None: ...
def _split_autocast(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    split_autocast creates a new graph module that splits the input graph module into multiple submodules
    based on the `_enter_autocast` and `_exit_autocast` nodes. It doesn't mutate the input graph module.

    Nodes between the **outer-most** `_enter_autocast` and `_exit_autocast(_enter_autocast)` are splitted
    into a submodule. Nested autocast regions are not splitted.
    `_enter_autocast` and `_exit_autocast(_enter_autocast)` nodes are in the submodule as well.

    Below is an example of splitting. A, B, C, D, E are blocks of non-autocast nodes in the original graph
    module. Nodes marked with the same number are grouped into the same submodule.
    A               # 0
    enter_autocast  # 1
    B               # 1
    exit_autocast   # 1
    C               # 2
    enter_autocast  # 3
    D               # 3
    exit_autocast   # 3
    E               # 4
    """
def _sequential_split_and_maybe_inline_subgraphs(gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replace_autocast_with_hop_pass().
    Split the graph module into multiple subgraphs based on the autocast nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    Nodes between `_enter_autocast` and `_exit_autocast(_enter_autocast)` are considered
    as a subgraph.
    """
def replace_autocast_with_hop_pass(gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
