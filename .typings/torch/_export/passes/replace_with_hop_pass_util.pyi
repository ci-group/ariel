import torch
from ..utils import node_replace_ as node_replace_, nodes_map as nodes_map
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch.export.graph_signature import ExportGraphSignature as ExportGraphSignature
from typing import Callable

def _replace_with_hop_helper(node: torch.fx.Node, enter_block_node: torch.fx.Node, wrap_hoo: HigherOrderOperator) -> None: ...
def _sequential_split_and_maybe_inline_subgraphs_helper(new_gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None, maybe_inline_or_replace_with_hop: Callable[[torch.fx.Node], None]) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replacing graph nodse with higher order nodes.
    For each subgraph in `new_gm`, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module, depending on `maybe_inline_or_replace_with_hop`.
    """
def _replace_with_hop_pass_helper(gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None, sequential_split_and_maybe_inline_subgraphs: Callable[[torch.fx.GraphModule, ExportGraphSignature | None], tuple[torch.fx.GraphModule, ExportGraphSignature | None]]) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
