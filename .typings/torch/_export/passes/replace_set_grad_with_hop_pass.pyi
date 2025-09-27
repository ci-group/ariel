import torch
from ..utils import node_inline_ as node_inline_, nodes_filter as nodes_filter, nodes_first as nodes_first, nodes_map as nodes_map, sequential_split as sequential_split
from .replace_with_hop_pass_util import _replace_with_hop_helper as _replace_with_hop_helper, _replace_with_hop_pass_helper as _replace_with_hop_pass_helper, _sequential_split_and_maybe_inline_subgraphs_helper as _sequential_split_and_maybe_inline_subgraphs_helper
from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled as wrap_with_set_grad_enabled
from torch.export.graph_signature import ExportGraphSignature as ExportGraphSignature

def _is_set_grad_enabled_node(node: torch.fx.Node) -> torch.fx.Node | bool: ...
def _is_set_grad_enabled_sub_mod(node: torch.fx.Node, omit_if_same_with_ambient: bool = False) -> bool | torch.Tensor: ...
def _replace_with_hop(node: torch.fx.Node) -> None: ...
def _remove_set_grad_and_inline(node: torch.fx.Node) -> None: ...
def _sequential_split_and_maybe_inline_subgraphs(gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replace_set_grad_with_hop_pass().
    Split the graph module into multiple subgraphs based on the set_grad_enabled nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    """
def replace_set_grad_with_hop_pass(gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
