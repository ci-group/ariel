import pickle
import torch
import torch.fx
from .graph_utils import _get_flat_args_unique as _get_flat_args_unique
from .symbolic_convert import InstructionTranslatorBase as InstructionTranslatorBase
from _typeshed import Incomplete
from collections import deque
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._pytree import tree_flatten as tree_flatten
from typing import Any, Callable, TypeVar

T = TypeVar('T')
Node = torch.fx.Node
Region = list[Node]
IdenticalNodes = list[Node]
GlobalStateKey = tuple[bool, bool, int, bool, bool, torch.dtype, bool, bool, bool, bool]
log: Incomplete
graph_expansion_log: Incomplete

def debug_log(msg: str, *args) -> None: ...
def _extract_tensor_metadata_for_node_hash(x: torch.Tensor) -> tuple[Callable[[T], T], tuple[Any, ...]]: ...

class NodeHashException(Exception): ...

class InputPickler(pickle.Pickler):
    _stream: Incomplete
    dispatch_table: Incomplete
    fast: bool
    def __init__(self) -> None: ...
    def dumps(self, obj: Any) -> bytes:
        """
        Pickle an object and return a byte string.
        """

def _extract_args(arg: Any) -> Any: ...
def _normalize_args(node: Node) -> tuple[tuple[str, ...], tuple[Any | None, ...]]: ...
def get_global_state_key() -> GlobalStateKey: ...

class BackwardBfsArgIter:
    _cur: Node | None
    _queue: deque[Node | None]
    def __init__(self, origin: Node) -> None: ...
    @staticmethod
    def create(origin: Node) -> BackwardBfsArgIter: ...
    def next(self) -> Node | None: ...
    def peek(self) -> Node | None: ...
    def add_children(self, node: Node) -> None: ...
    def _append(self, arg: Node) -> None: ...
    def __str__(self) -> str: ...

class GraphRegionTracker:
    """
    GraphRegionTracker tracks each node added to the output graph and generates a key based on the source location,
    instruction pointer, input shapes, and global state at the time the node is inserted into the graph. Nodes with
    the same key are grouped together in a list of identical nodes (the value of node_to_duplicates).

    hash_to_duplicates: Dict[str, IdenticalNodes] - A dictionary mapping the key to a list of identical nodes
    node_to_duplicates: Dict[Node, IdenticalNodes] - A dictionary mapping a node to the list of identical nodes it belongs to
    input_pickler: InputPickler - An instance of InputPickler used to generate a node hash
    """
    hash_to_duplicates: dict[str, IdenticalNodes]
    node_to_duplicates: dict[Node, IdenticalNodes]
    node_to_mutated_arg_positions: dict[Node, OrderedSet[int]]
    input_pickler: Incomplete
    def __init__(self) -> None: ...
    def _hash_node(self, filename: str, lineno: int, instruction_pointer: int | None, node: Node) -> str: ...
    def _is_identical(self, n0: Node, n1: Node) -> bool: ...
    def track_node(self, tx: InstructionTranslatorBase, node: Node) -> None:
        """
        The main entry point for tracking a node. This function will hash the node argument and group
        nodes with the same hash together. It updates the hash_to_duplicates and node_to_duplicates dictionaries
        to track the new node.
        """
    def track_node_mutations(self, node: Node, flat_args_kwargs: list[Any], id_to_initial_version: dict[int, int]) -> None:
        """
        This function tracks which argument positions are mutated by the given node. Subgraph HOP does not support
        input mutations today so we will skip regions which have inputs that are mutated.
        """
    def add_node_mutation(self, node: Node, arg_pos: int) -> None: ...
    def get_identical_regions(self, graph: torch.fx.Graph) -> list[list[Region]]:
        """
        This function is responsible for extracting the largest regions of identical nodes from the given graph.
        **Note**: This function assumes the nodes that have been tracked with track_node are in the provided graph argument.

        The algorithm proceeds as follows:
        The nodes tracked via track_node above are organized into region groups. The initial region groups look like this:
        [[IdenticalNode1], [IdenticalNode2], [IdenticalNode3]] and each sublist is called a region. For each region group
        (starting at the topologically latest region group), the inner regions are gradually expanded one node at time from
        the flattened args and kwargs of the node in each region provided that for all regions in the group, the nodes being
        added are also identical (ie have the same key computed by track_node). This is checked by verifying that the two
        nodes have the same identical node list in node_to_duplicates.
        """
    def __str__(self) -> str: ...

class RegionWrapper:
    """Holds state for regions e.g. ancestors and new candidate nodes for consideration"""
    node_to_recursive_ancestors: Incomplete
    iter: Incomplete
    nodes_unique: Incomplete
    ancestors: Incomplete
    region: Incomplete
    def __init__(self, region: Region, node_to_recursive_ancestors: dict[Node, set[Node]]) -> None: ...
    def next_candidate(self) -> Node | None: ...
    def will_inclusion_create_cycle(self, node: Node) -> bool: ...
    def add(self, node: Node) -> None: ...

def fully_expand_region_group(regions: list[Region], seen_nodes: set[Node], node_to_recursive_ancestors: dict[Node, set[Node]], is_identical_fn: Callable[[Node, Node], bool]) -> None: ...
def _populate_recursive_ancestor_map(graph: torch.fx.Graph) -> dict[Node, set[Node]]: ...
