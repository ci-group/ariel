import networkx as nx
from _typeshed import Incomplete
from torch.fx import Graph as Graph, Node as Node
from typing import Any

class GraphInfoProvider:
    """
    This class provides information about the graph, such as the nodes, edges, and their runtime and memory requirements.
    It also provides methods to create graphs from the information provided.
    """
    __RECOMPUTABLE_NODE_ONLY_GRAPH: str
    __RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT: str
    __FULL_NX_JOINT_GRAPH: str
    __SIMPLIFIED_FX_JOINT_GRAPH: str
    graph_nodes_in_order: Incomplete
    graph_edges: Incomplete
    all_node_runtimes: dict[str, float]
    all_node_memories: dict[str, float]
    all_recomputable_banned_nodes: Incomplete
    all_recomputable_banned_nodes_set: Incomplete
    recorded_knapsack_input_memories: Incomplete
    recorded_knapsack_input_runtimes: Incomplete
    _lazily_initialized_graphs: dict[str, Any]
    def __init__(self, graph_nodes_in_order: list[str], graph_edges: list[tuple[str, str]], all_recomputable_banned_nodes: list[str], all_node_runtimes: dict[str, float] | None = None, all_node_memories: dict[str, float] | None = None, recorded_knapsack_input_memories: list[float] | None = None, recorded_knapsack_input_runtimes: list[float] | None = None, joint_graph: Graph | None = None) -> None: ...
    @classmethod
    def inialize_from_graph(cls, joint_graph: Graph, all_recomputable_banned_nodes: list[Node], recorded_knapsack_input_memories: list[float], recorded_knapsack_input_runtimes: list[float]) -> GraphInfoProvider:
        """
        Enables initialization from a joint graph.
        """
    @property
    def recomputable_node_only_graph(self) -> nx.DiGraph: ...
    @property
    def recomputable_node_only_graph_with_larger_graph_context(self) -> nx.DiGraph: ...
    @property
    def full_joint_nx_graph(self) -> nx.DiGraph: ...
    @property
    def simplified_fx_joint_graph(self) -> Graph: ...
    def get_non_ac_peak_memory(self) -> float: ...
    def get_theoretical_max_runtime(self) -> float: ...
    def get_knapsack_memory_input(self) -> list[float]: ...
    def get_knapsack_runtime_input(self) -> list[float]: ...
    def _create_recomputable_node_only_graph(self) -> nx.DiGraph: ...
    def _create_recomputable_node_only_graph_with_larger_graph_context(self) -> nx.DiGraph: ...
    def _create_full_joint_graph(self) -> nx.DiGraph: ...
    def _recreate_psuedo_joint_graph(self) -> Graph: ...
    def _visualize_recomputable_candidate_graph_with_larger_context(self, layout_k: float = 0.5, layout_iterations: int = 30) -> None:
        """
        Visualize the recomputable candidate graph with larger context.
        """
