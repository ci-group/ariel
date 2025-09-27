import networkx as nx
from _typeshed import Incomplete
from torch._functorch._activation_checkpointing.graph_info_provider import GraphInfoProvider as GraphInfoProvider
from typing import Callable

class KnapsackEvaluator:
    """
    This class evaluates the theoretical runtime and peak memory usage of a given checkpointing strategy.
    It takes in a graph and a list of nodes that are saved and recomputed, and then simulates the
    backward pass to calculate the peak memory usage.
    """
    _graph_info_provider: Incomplete
    def __init__(self, graph_info_provider: GraphInfoProvider) -> None: ...
    def _get_backward_memory_from_topologically_sorted_graph(self, node_graph: nx.DiGraph, node_memories: dict[str, float], saved_nodes_set: set[str], peak_memory_after_forward_pass: float) -> list[tuple[float, str]]:
        """
        Simulates the backward pass and keeps track of the peak memory usage.

        High Level Steps:
            1. Set Initial Peak/Current Memory
                Allows you to set the peak memory after the forward pass, but typically this is
                the sum of the estimated memory of the saved nodes.
            2. Perform a reverse topological sort of the node_graph.
                If full graph is defined then will sort the full graph and only process the subset
                of nodes in the node_graph.
            3. Iterate through the sorted graph nodes.
                If the node is saved then just drop it's memory from current memory.
                If the node is not saved then add it's memory to current memory and then traverse it's
                predecessors to simulate recomuptation chain. Will check if new peak memory after all
                predecessors are processed.

        Args:
            node_graph (nx.DiGraph): A directed graph representing the recomputable forward nodes.
            saved_nodes_set (Set[str]): A set of node names that are saved.
            peak_memory_after_forward_pass (float): The peak memory usage after the forward pass.
        """
    def _validate_all_indexes_accounted_for_in_provided_output(self, saved_nodes_idxs: list[int], recomputable_node_idxs: list[int]) -> None:
        """
        Validate that all indexes are accounted for in the provided output.
        This function checks that the union of saved nodes and recomputable nodes
        covers all candidate nodes without any overlaps.
        """
    def evaluate_knapsack_output(self, saved_nodes_idxs: list[int], recomputable_node_idxs: list[int], account_for_backward_pass: bool = False) -> dict[str, float]:
        """
        Evaluate the theoretical runtime and peak memory usage of a given checkpointing strategy.
        Args:
        - saved_nodes_idxs (List[int]): The indices of nodes that are saved.
        - recomputable_node_idxs (List[int]): The indices of nodes that need to be recomputed.
        """
    def evaluate_distribution_of_results_for_knapsack_algo(self, knapsack_algo: Callable[[list[float], list[float], float], tuple[float, list[int], list[int]]], memory_budget_values: list[float]) -> list[dict[str, float]]:
        """
        Evaluates the distribution of results for a given knapsack algorithm.
        Args:
            knapsack_algo (Callable): The knapsack algorithm to use for evaluation.
            memory_budget_values (List[float]): A list of memory budgets to evaluate.
        """
    def get_knee_point_memory_budget(self, knapsack_algo: Callable[[list[float], list[float], float], tuple[float, list[int], list[int]]], max_mem_budget: float = 0.1, min_mem_budget: float = 0.001, iterations: int = 100) -> float:
        """
        Finds the memory budget at the knee point in the Pareto frontier.

        The knee point is defined as the point where the trade-off between
        runtime and memory usage is optimal.

        Args:
            knapsack_algo (callable): Knapsack algorithm to use for evaluation.
            max_mem_budget (float, optional): Maximum memory budget. Defaults to 0.1.
            min_mem_budget (float, optional): Minimum memory budget. Defaults to 0.001.
            iterations (int, optional): Number of memory budgets to evaluate. Defaults to 100.

        Returns:
            float: Memory budget at the knee point.
        """
