import torch
from .graph_module import _is_observed_standalone_module as _is_observed_standalone_module
from .quantize_handler import QuantizeHandler as QuantizeHandler
from torch.ao.quantization.qconfig import QConfigAny as QConfigAny
from torch.ao.quantization.utils import MatchAllNode as MatchAllNode, Pattern as Pattern
from torch.fx.graph import Graph as Graph, Node as Node
from torch.nn.utils.parametrize import type_before_parametrizations as type_before_parametrizations
from typing import Any, Callable

__all__: list[str]
_MatchResult = tuple[Node, list[Node], Pattern | None, QuantizeHandler]
_MatchResultWithQConfig = tuple[Node, list[Node], Pattern | None, QuantizeHandler, QConfigAny]

def _is_match(modules, node, pattern, max_uses=...):
    """Matches a node in fx against a pattern"""
def _find_matches(graph: Graph, modules: dict[str, torch.nn.Module], patterns: dict[Pattern, QuantizeHandler], root_node_getter_mapping: dict[Pattern, Callable], standalone_module_names: list[str] | None = None, standalone_module_classes: list[type] | None = None, custom_module_classes: list[Any] | None = None) -> dict[str, _MatchResult]:
    """
    Matches the nodes in the input graph to quantization patterns, and
    outputs the information needed to quantize them in future steps.

    Inputs:
      - graph: an fx.Graph object
      - modules: a mapping of fully qualified module name to instance,
          for example, {'foo': ModuleFoo, ...}
      - patterns: a mapping from a tuple of nodes in reverse order to
          uninitialized QuantizeHandler subclass.

    Outputs a map of
      node_name ->
        (node, matched_values, matched_pattern, QuantizeHandler instance,
         qconfig)

    For example, {
      'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,
                 <CopyNodeQuantizeHandler instance>, QConfig(...)),
      ...
    }
    """
