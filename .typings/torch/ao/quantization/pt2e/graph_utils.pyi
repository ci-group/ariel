import torch
from torch.export import ExportedProgram
from torch.fx import Node
from typing import Any, Callable

__all__ = ['find_sequential_partitions', 'get_equivalent_types', 'update_equivalent_types_dict', 'bfs_trace_with_node_process']

def get_equivalent_types() -> list[set]: ...
def update_equivalent_types_dict(customized_equivalent_types=None) -> None:
    """Help function for user who wants to customize the _EQUIVALENT_TYPES and _EQUIVALENT_TYPES_DICT.
    When customized_equivalent_types passes in,
    re-generate _EQUIVALENT_TYPES and _EQUIVALENT_TYPES_DICT.
    """
def find_sequential_partitions(gm: torch.fx.GraphModule, partition_types: list[Any], include_functional_equivalent: bool = True, filter_fn: Callable[[Node], bool] | None = None): ...
def bfs_trace_with_node_process(model: ExportedProgram | torch.fx.GraphModule, node_op: Callable) -> None:
    """Traverse the graph module and apply node_op to each node."""
