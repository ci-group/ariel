from _typeshed import Incomplete
from collections.abc import Generator

class DiGraph:
    """Really simple unweighted directed graph data structure to track dependencies.

    The API is pretty much the same as networkx so if you add something just
    copy their API.
    """
    _node: Incomplete
    _succ: Incomplete
    _pred: Incomplete
    _node_order: Incomplete
    _insertion_idx: int
    def __init__(self) -> None: ...
    def add_node(self, n, **kwargs) -> None:
        """Add a node to the graph.

        Args:
            n: the node. Can we any object that is a valid dict key.
            **kwargs: any attributes you want to attach to the node.
        """
    def add_edge(self, u, v) -> None:
        """Add an edge to graph between nodes ``u`` and ``v``

        ``u`` and ``v`` will be created if they do not already exist.
        """
    def successors(self, n):
        """Returns an iterator over successor nodes of n."""
    def predecessors(self, n):
        """Returns an iterator over predecessors nodes of n."""
    @property
    def edges(self) -> Generator[Incomplete]:
        """Returns an iterator over all edges (u, v) in the graph"""
    @property
    def nodes(self):
        """Returns a dictionary of all nodes to their attributes."""
    def __iter__(self):
        """Iterate over the nodes."""
    def __contains__(self, n) -> bool:
        """Returns True if ``n`` is a node in the graph, False otherwise."""
    def forward_transitive_closure(self, src: str) -> set[str]:
        """Returns a set of nodes that are reachable from src"""
    def backward_transitive_closure(self, src: str) -> set[str]:
        """Returns a set of nodes that are reachable from src in reverse direction"""
    def all_paths(self, src: str, dst: str):
        """Returns a subgraph rooted at src that shows all the paths to dst."""
    def first_path(self, dst: str) -> list[str]:
        """Returns a list of nodes that show the first path that resulted in dst being added to the graph."""
    def to_dot(self) -> str:
        """Returns the dot representation of the graph.

        Returns:
            A dot representation of the graph.
        """
