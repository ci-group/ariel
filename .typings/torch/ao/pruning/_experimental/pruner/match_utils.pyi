from torch import nn as nn
from torch.ao.quantization.utils import MatchAllNode as MatchAllNode
from torch.fx import Node as Node
from torch.nn.utils import parametrize as parametrize
from typing import Any

def _match(modules: dict[str, nn.ModuleDict], node: Node, current: nn.Module | Any) -> bool:
    """
    checks to see if a single node of a pattern matches
    """
def apply_match(modules: dict[str, nn.ModuleDict], pattern: tuple[Any] | Any, node: Node, matched_node_pattern: list[Node]) -> list[Node] | None:
    """
    This function will return the matched nodes if the pattern matches the node given
    If there is no match, it will return None
    """
