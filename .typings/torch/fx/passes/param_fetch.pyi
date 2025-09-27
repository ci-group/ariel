import torch.nn as nn
from torch.fx.graph_module import GraphModule
from typing import Any

__all__ = ['default_matching', 'extract_attrs_for_lowering', 'lift_lowering_attrs_to_nodes']

def default_matching(name: str, target_version: int) -> str:
    """Default matching method"""
def extract_attrs_for_lowering(mod: nn.Module) -> dict[str, Any]:
    """If `mod` is in `module_fetch_book`, fetch the mod's attributes that in the `module_fetch_book`
    after checking module's version is compatible with the `module_fetch_book`.
    """
def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None:
    """Recursively traverse all `fx_module` nodes and fetch the module's attributes if the node is a leaf module."""
