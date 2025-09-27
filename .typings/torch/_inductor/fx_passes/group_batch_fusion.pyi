import collections
import torch
from .. import config as config
from ..pattern_matcher import CallFunctionVarArgs as CallFunctionVarArgs, get_arg_value as get_arg_value, stable_topological_sort as stable_topological_sort
from ..utils import OPTIMUS_EXCLUDE_POST_GRAD as OPTIMUS_EXCLUDE_POST_GRAD
from _typeshed import Incomplete
from collections.abc import Iterable, Iterator
from torch._dynamo.utils import counters as counters, is_node_meta_valid as is_node_meta_valid
from torch._logging import trace_structured as trace_structured
from torch.fx.passes.graph_transform_observer import GraphTransformObserver as GraphTransformObserver
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any

has_fbgemm: bool
aten: Incomplete
log: Incomplete
DEFAULT_BETA: int
DEFAULT_ALPHA: int
MIN_FUSE_SET_SIZE: int
MAX_FUSE_SET_SIZE: int
MAX_FUSE_SEARCH_DEPTH: int
MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR: int
FUSE_NODES_WITH_SAME_PARENT: bool
SHAPE_BROADCAST_BATCH_LINEAR: bool
Fuse_NODES_WITH_SAME_USERS: bool
SEARCH_EXCLUSIONS: Incomplete
default_graph_search_options: Incomplete
graph_search_options = default_graph_search_options

def update_stack_example_value(node, metadata, dim: int = 0, op=...) -> None:
    """
    Update the example value of the node in the graph to enable followup split cat opt.
    """
def update_pointwise_example_value(pointwise_node, input, other, op) -> None:
    """
    Update the example value of the add node in the graph to enable followup split cat opt.
    """

class GroupBatchFusionBase:
    graph_search_options: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def match(self, node) -> None: ...
    def fuse(self, graph, subset) -> None: ...

PRE_GRAD_FUSIONS: dict[str, GroupBatchFusionBase]
POST_GRAD_FUSIONS: dict[str, GroupBatchFusionBase]

def register_fusion(name: str, pre_grad: bool = True): ...
def list_group_batch_fusions(pre_grad: bool = True) -> list[str]: ...
def decompose_stack(graph: torch.fx.GraphModule, input_tensors: list[Any]) -> Any: ...

class GroupFusion(GroupBatchFusionBase):
    """
    Fuse ops in a group way, e.g, fuse mm/addmm of arbitrary input shapes with fbgemm.gmm.
    """
class BatchFusion(GroupBatchFusionBase):
    """
    Fuse ops in a batch way, e.g, fuse mm/addmm of same input shapes with bmm.
    """

class BatchPointwiseOpsFusionFactory(BatchFusion):
    op: Incomplete
    def __init__(self, op, **kwargs) -> None: ...

class PostGradBatchLinearFusion(BatchFusion):
    """
    Fuse ops in a batch way in post grad (aten level).
    """
    def _addmm_node_can_be_fused(self, node: torch.fx.Node) -> bool: ...
    def _is_input_2d(self, input: torch.fx.Node) -> bool: ...
    def match(self, node: torch.fx.Node) -> tuple[str, int, int, int, bool, str] | None: ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

class GroupLinearFusion(GroupFusion):
    def _addmm_node_can_be_fused(self, node: torch.fx.Node): ...
    def _mm_node_can_be_fused(self, node: torch.fx.Node): ...
    def match(self, node: torch.fx.Node) -> tuple[str, bool] | None: ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

class BatchPointwiseMathOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise math operator (e.g., add, mul) in post grad pass.
    """
    op: Incomplete
    def __init__(self, op, **kwargs) -> None: ...
    def _pointwise_node_can_be_fused(self, node: torch.fx.Node): ...
    def match(self, node: torch.fx.Node): ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

class BatchLinearLHSFusion(BatchFusion):
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """
    def match(self, node: torch.fx.Node) -> tuple[str, bool, Any] | None: ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

def _is_mutable_node(tgt): ...
def is_linear_node_can_be_fused(node: torch.fx.Node): ...

class PreGradBatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddmm
    """
    def _getitem_args(self, getitem_node: torch.fx.Node): ...
    def match(self, node: torch.fx.Node): ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

class BatchLayernormFusion(BatchFusion):
    """
    Batch layer norm fusion in pre grad pass
    """
    def match(self, node: torch.fx.Node): ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """
    op: Incomplete
    def __init__(self, op, **kwargs) -> None: ...
    def match(self, node: torch.fx.Node): ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

class BatchPointwiseOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in post grad pass.
    The introduced stack node may be merged in split cat.
    """
    op: Incomplete
    def __init__(self, op, **kwargs) -> None: ...
    def match(self, node: torch.fx.Node): ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

class BatchMathOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch simple match related ops such as nan_to_num in pre grad pass.
    """
    op: Incomplete
    def __init__(self, op, **kwargs) -> None: ...
    def match(self, node: torch.fx.Node): ...
    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]): ...

class BatchTanhPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchSigmoidPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchReLuPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchDetachPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchNanToNumPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchClampPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchTanhPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchSigmoidPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchReLuPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchAddPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchSubPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchDivPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None: ...

class BatchMulPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None: ...

class _OrderedSet:
    rep: Incomplete
    def __init__(self, param=None) -> None: ...
    def __contains__(self, o) -> bool: ...
    def __len__(self) -> int: ...
    def append(self, o) -> None: ...
    def __iter__(self): ...

def find_independent_subset_greedy(node_list: Iterable[torch.fx.Node], graph_search_options: dict[str, Any]) -> Iterator[Iterable[torch.fx.Node]]:
    """
    Yields a list of subsets of `node_list` where no element in the subset
    depends on any other element in the subset. This results in a set of
    independent nodes which can be fused together.

    The order of `node_list` is preserved within each subset so we can benefit
    from split-cat elimination in later passes.

    During iteration it is only safe to mutate the graph by changing the nodes
    that have been returned.

    graph_search_options:
      - min_fuse_set_size: Minimum size of the subset to consider. Subsets below
        this size will be ignored.
      - max_fuse_set_size: Maximum size of the subset to consider. Subsets will
        be broken to be at most this size.
    """
def get_fusion_candidates(rule: GroupBatchFusionBase, root_node: torch.fx.Node, fused_set: OrderedSet[torch.fx.Node]) -> collections.defaultdict[Any, list[torch.fx.Node]]:
    '''
    Search fusion candidates for a specific rule using BFS starting from the root node.
    We only search the subgraph within graph_search_options["max_fuse_search_depth"].
    '''
def apply_group_batch_fusion(graph: torch.fx.GraphModule, rule: GroupBatchFusionBase): ...
def generate_fusion_from_config(config_options: dict[str, Any], pre_grad: bool = True): ...
def group_batch_fusion_passes(graph: torch.fx.Graph, pre_grad: bool = True): ...
