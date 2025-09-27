import torch
from .. import config as config, inductor_prims as inductor_prims
from ..pattern_matcher import CallFunction as CallFunction, Ignored as Ignored, KeywordArg as KeywordArg, ListOf as ListOf, MULTIPLE as MULTIPLE, Match as Match, PatternExpr as PatternExpr, PatternMatcherPass as PatternMatcherPass
from _typeshed import Incomplete
from dataclasses import dataclass, field
from torch.utils._ordered_set import OrderedSet as OrderedSet

log: Incomplete
aten: Incomplete
patterns: Incomplete

def _is_backward(graph: torch.fx.Graph) -> bool: ...
def _compute_mm_arithmetic_intensity(M: int, N: int, K: int) -> float: ...
def _filter_nodes_by_target(nodes: list[torch.fx.Node], target) -> list[torch.fx.Node]: ...
def _find_ancestors(node: torch.fx.Node) -> OrderedSet[torch.fx.Node]: ...
def _get_tensor(node: torch.fx.Node) -> torch.Tensor: ...

@dataclass
class _AllGatherMatch:
    match: Match
    shard_node: torch.fx.Node
    ag_node: torch.fx.Node
    res_node: torch.fx.Node
    gather_dim: int
    group_name: str
    def replace_with(self, new_node: torch.fx.Node) -> None: ...
    def erase(self) -> None: ...

def find_all_gather_patterns(graph: torch.fx.Graph): ...

@dataclass
class _ReduceScatterMatch:
    match: Match
    input_node: torch.fx.Node
    reduce_scatter_node: torch.fx.Node
    wait_tensor_node: torch.fx.Node
    reduce_op: str
    scatter_dim: int
    group_name: str
    def replace_with(self, new_node: torch.fx.Node) -> None: ...
    def _update_save_for_backward(self, new_node: torch.fx.Node) -> None:
        """
        If the output node is a user of the reduce_scatter node (indicating the reduce_scatter
        result is saved for backward), this method will update the output node to use the fused node instead.
        """
    def erase(self) -> None: ...

def find_reduce_scatter_patterns(graph: torch.fx.Graph): ...

@dataclass
class _Matmul:
    nodes: list[torch.fx.Node]
    arg_ancestor_nodes: OrderedSet[torch.fx.Node] = field(init=False)
    A_node: torch.fx.Node
    B_node: torch.fx.Node
    pre_mm_reshape: torch.fx.Node | None
    post_mm_reshape: torch.fx.Node | None
    def __post_init__(self) -> None: ...
    def replace_with(self, new_node: torch.fx.Node) -> None:
        """
        Replace the matmul with the new node.
        """
    def erase(self) -> None: ...
    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> _Matmul: ...

@dataclass
class _ScaledMatmul(_Matmul):
    A_scale_node: torch.fx.Node
    B_scale_node: torch.fx.Node
    bias_node: torch.fx.Node | None
    result_scale_node: torch.fx.Node | None
    out_dtype: torch.dtype | None
    use_fast_accum: bool
    pre_mm_reshape: torch.fx.Node | None
    post_mm_reshape: torch.fx.Node | None
    def __post_init__(self) -> None: ...
    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> _ScaledMatmul: ...

def _find_reshape_mm_reshape(node: torch.fx.Node) -> list[_Matmul]: ...
def _find_consumer_matmuls(node: torch.fx.Node) -> list[_Matmul]:
    """
    Find the matmuls that use `node` as the lhs argument.
    """
def _insert_fused_all_gather_matmul(graph: torch.fx.Graph, matmuls: list[_Matmul], shard_node: torch.fx.Node, gather_dim: int, group_name: str) -> torch.fx.Node: ...
def fuse_all_gather_matmul(all_gather: _AllGatherMatch) -> None:
    """
    Fused the pattern

        A = all_gather_tensor(A_shard, gather_dim, group_name)
        C_0 = torch.matmul(A, B_0)
        C_1 = torch.matmul(A, B_1)
        C_2 = torch.matmul(A, B_2)
        ...

    into

        A, Cs = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, [B_0, B_1, B_2, ...], gather_dim, group_name,
        )
    """
def _scatter_dim_after_reshape(reshape_node: torch.fx.Node, orig_scatter_dim: int) -> int:
    """
    Given a reshape node and the original scatter dim for the target tensor,
    returns the new scatter dim for the reshaped tensor.
    """
def _find_producer_matmul(node: torch.fx.Node) -> _Matmul | None:
    """
    Returns producer matmul node if found, otherwise returns None.
    """
def _insert_fused_matmul_reduce_scatter(graph: torch.fx.Graph, matmul: _Matmul, reduce_op: str, orig_scatter_dim: int, group_name: str, scatter_dim_after_reshape: int, output_shape: list[int]) -> torch.fx.Node: ...
def fuse_matmul_reduce_scatter(reduce_scatter: _ReduceScatterMatch) -> None:
    """
    Fused the pattern

        reduce_scatter_tensor(A @ B, scatter_dim, group_name)

    into

        torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, scatter_dim, group_name,
        )

    Returns boolean indicating if fusion was successful or not.
    """
def _get_node_to_ancestors(graph: torch.fx.Graph) -> dict[torch.fx.Node, OrderedSet[torch.fx.Node]]:
    """
    Compute the ancestors for all nodes in a graph.
    """
def _get_collective_to_overlappable_nodes(graph: torch.fx.Graph) -> dict[torch.fx.Node, list[torch.fx.Node]]:
    """
    For each collective in the graph, find nodes that are neither ancestors nor
    descendants of the collective.
    """
def _get_unexposed_collectives(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    '''
    Find all unexposed collectives in the graph.

    Because we don\'t have the runtime estimate, this function is a rough
    estimation using the following strong/hand-wavy assumptions:

    - Only a predefined set of "compute intensive" operation can hide a collective.
    - Any "compute intensive" operation can hide exactly one collective.
    '''
def micro_pipeline_tp_pass(graph: torch.fx.Graph): ...
