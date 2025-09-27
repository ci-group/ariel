from dataclasses import dataclass, field
from torch.fx.graph import Graph
from torch.fx.node import Node
from typing import Any, Callable

__all__ = ['get_source_partitions', 'check_subgraphs_connected', 'SourcePartition']

@dataclass
class SourcePartition:
    nodes: list[Node]
    source: Any
    input_nodes: list[Node] = field(default_factory=list)
    output_nodes: list[Node] = field(default_factory=list)
    params: list[Node] = field(default_factory=list)

def get_source_partitions(graph: Graph, wanted_sources: list[Any], filter_fn: Callable[[Node], bool] | None = None) -> dict[Any, list[SourcePartition]]:
    """
    Args:
        graph: The graph we want to partition
        wanted_sources: List of sources of nodes that were decomposed from this
            source. This can be a function (ex. torch.nn.functional.linear) or a
            leaf module type (ex. torch.nn.Linear).

    Returns:
        Dictionary mapping sources that were given to a list of SourcePartitions
        that correspond to the list of nodes that were decomposed from the given
        source.
    """
def check_subgraphs_connected(subgraph1: SourcePartition, subgraph2: SourcePartition) -> bool:
    """
    Given two subgraphs A and B (in the form of a list of nodes), checks if
    A has nodes connecting to at least one node in B -- aka there exists a node
    in B that uses a node in A (not the other way around).
    """
