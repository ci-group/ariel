from .ns_types import NSNodeTargetType as NSNodeTargetType
from _typeshed import Incomplete
from torch.ao.quantization import FakeQuantizeBase as FakeQuantizeBase, ObserverBase as ObserverBase
from torch.ao.quantization.backend_config import get_native_backend_config as get_native_backend_config
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers as _get_pattern_to_quantize_handlers
from torch.ao.quantization.utils import getattr_from_fqn as getattr_from_fqn
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node
from typing import Any, Callable

toq: Incomplete

def get_type_a_related_to_b(base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]]) -> set[tuple[NSNodeTargetType, NSNodeTargetType]]: ...
NSFusionElType = Callable | str | tuple[str, Any]
NSFusionType = tuple[NSFusionElType, NSFusionElType] | tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType]

def get_reversed_fusions() -> list[tuple[NSFusionType, int]]:
    """
    Set of potential fusions, in reverse order.  The order is reversed
    to match how fusion patterns are defined in quantization code.

    Fusion format:
    ((fusion_op_0, fusion_op_1), base_op_idx)

    Where base_op_idx is the idx of the op we should use to match other related
    ops. Note: base_op_idx is specified in non-reverse order, i.e. a base_op_idx
    of 0 represents the first op in regular (non-reverse) order, 1 represents the
    second op, etc.
    """
def end_node_matches_reversed_fusion(end_node: Node, reversed_fusion: NSFusionType, gm: GraphModule, seen_nodes: set[Node]) -> bool:
    """
    Returns true if a pattern ending with `end_node` matches
    the fusion pattern.
    """
