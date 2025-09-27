import torch
from .. import config as config
from ..pattern_matcher import Arg as Arg, CallFunction as CallFunction, KeywordArg as KeywordArg, MULTIPLE as MULTIPLE, Match as Match, PatternMatcherPass as PatternMatcherPass, init_once_fakemode as init_once_fakemode, register_graph_pattern as register_graph_pattern, stable_topological_sort as stable_topological_sort
from .decompose_mem_bound_mm import check_device as check_device
from .replace_random import replace_random_passes as replace_random_passes
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._dynamo.utils import counters as counters
from torch._inductor.constant_folding import ConstantFolder as ConstantFolder
from torch._inductor.fx_passes.dedupe_symint_uses import _SymHashingDict as _SymHashingDict
from torch._inductor.utils import get_gpu_type as get_gpu_type
from torch.fx.experimental.symbolic_shapes import guard_or_false as guard_or_false, guard_or_true as guard_or_true, statically_known_true as statically_known_true
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any

log: Incomplete
patterns: Incomplete
aten: Incomplete
prims: Incomplete
pass_patterns: Incomplete

@init_once_fakemode
def lazy_init() -> None: ...
def remove_no_ops(gm: torch.fx.GraphModule, zeros: OrderedSet[torch.fx.Node], ones: OrderedSet[torch.fx.Node]): ...
def remove_redundant_views(gm: torch.fx.GraphModule):
    """
    Removes redundant views by reusing existing ones.
    """

class UniformValueConstantFolder(ConstantFolder):
    """
    Runs constant folding and replaces tensors that have a uniform value
    with a tensor constructor call: aten.full([shape], value, ...)
    """
    node_storages_ptrs: dict[torch.fx.Node, int]
    constant_data_ptrs: dict[torch.fx.Node, StorageWeakRef]
    node_replacements_shapes: dict[torch.fx.Node, list[int]]
    symint_nodes: Incomplete
    view_op_packets: Incomplete
    indexing_op_packets: Incomplete
    def __init__(self, gm, skip_constructors: bool = False) -> None: ...
    def _add_peephole_patterns(self) -> None:
        """
        Add peephole patterns for nodes where we can infer constant value even if some inputs
        of the node are unknown.
        """
    def _support_dynamic_shape(self): ...
    def insertable_tensor_check(self, t: torch.Tensor) -> bool: ...
    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None: ...
    def insert_placerholder_values(self, env: dict[torch.fx.Node, Any]) -> None: ...
    def _deduce_value(self, node: torch.fx.Node): ...

def constant_fold_uniform_value(gm: torch.fx.GraphModule): ...
def canonicalize_quant_mapping(gm: torch.fx.GraphModule):
    """


    torch.ops.higher_order.invoke_quant_packed(repeated_subgraph0, 'quant_invoke_0_0', (arg0_1, arg1_1));
    ->
    torch.ops.higher_order.invoke_quant(repeated_subgraph0, arg0_1, arg1_1, scheme = 'nf4');
    """
def canonicalize_aten_ir_passes(gm: torch.fx.GraphModule):
    """
    Canonicalization passes that will run immediately after aot autograd
    tracing. Thsis must be run before all other graph passes.
    """
def joint_graph_passes(graph: torch.fx.GraphModule):
    """
    Run FX transformations on the joint forwards+backwards graph.
    """
def fix_iota_device(match: Match, length, start, step, dtype, device, requires_grad):
    '''
    Eager supports:

        aten.index(cuda_tensor, torch.arange(..., device="cpu"))

    But this results in an implicit host-device-copy and breaks cudagraphs.
    Rewrite the arange to use CUDA.
    '''
def pointless_convert(match: Match, arg, dtype1: torch.dtype, dtype2: torch.dtype):
    """Remove chain of dtype conversions often created by AMP"""
def definitely_equal(old_sizes: Sequence[torch.SymInt | int], new_sizes: Sequence[torch.SymInt | torch.fx.Node | int]) -> bool:
    """
    Leverage guard_or_true/false to compare if two lists of int/symint are equal.
    Useful to compare sizes, strides etc.

    Can handle -1 in new_sizes which happens in the size arguments of a
    view op. old_sizes is supposed to be the tensor shape and should not
    contain -1.

    new_sizes can contains fx.Node when dynamic shape is enabled. In that
    case new_sizes[i].meta['val'] contains the real torch.SymInt.
    """
def pointless_view(match: Match, arg, size):
    """Remove no-op view"""
def pointless_view_pair(match: Match, arg, size1, size2):
    """
    Remove a pair of views that are pointless.
    """
def pointless_permute_pair(match: Match, arg, perm1, perm2): ...
def bmm_to_mm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    """Convert bmm to mm when batch size is 1"""
def _partial_softmax_pattern(linear_func, reverse: bool = False, to_dtype: bool = False): ...
def _other_is_broadcasted_in_dim(match): ...
def mul_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None): ...
def div_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None): ...
