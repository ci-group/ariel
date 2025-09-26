import torch
from ..._dynamo.utils import counters as counters
from ..ir import ComputedBuffer as ComputedBuffer, FixedLayout as FixedLayout, FlexibleLayout as FlexibleLayout, InputBuffer as InputBuffer, StorageBox as StorageBox, Subgraph as Subgraph, TensorBox as TensorBox
from ..lowering import lowerings as lowerings
from ..pattern_matcher import Arg as Arg, CallFunction as CallFunction, Match as Match, PatternMatcherPass as PatternMatcherPass, register_graph_pattern as register_graph_pattern
from ..select_algorithm import ExternKernelChoice as ExternKernelChoice, SymbolicGridFn as SymbolicGridFn, TritonTemplate as TritonTemplate, TritonTemplateCaller as TritonTemplateCaller, autotune_select_algorithm as autotune_select_algorithm
from ..utils import ceildiv as ceildiv
from _typeshed import Incomplete
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._pytree import tree_map as tree_map

B2B_GEMM_PASS: Incomplete

@SymbolicGridFn
def b2b_gemm_grid(M, P, meta, *, cdiv): ...

b2b_gemm_left_template: Incomplete
b2b_gemm_right_template: Incomplete

def load_ratio_left(M: int, N: int, O: int, P: int, m: int, n: int, o: int, p: int) -> float:
    """
    compute the ratio of estimated numbers of loads in baseline and b2bgemm
    M, N, O, P are matrix sizes
    m, n, o, p are block sizes
    |       | baseline (lower bound)        | b2bgemm
    | load  | M * N + N * O + M * O + O * P | M / m * P / p * O / o * (o * p + N / n * (m * n + n * o))
    | store | M * O + M * P                 | M * P
    b2bgemm is always better on stores, but for loads we need to find out beneficial cases using this function
    """
def load_ratio_right(M: int, N: int, O: int, P: int, m: int, n: int, o: int, p: int) -> float:
    """
    compute the ratio of estimated numbers of loads in baseline and b2bgemm
    M, N, O, P are matrix sizes
    m, n, o, p are block sizes
    |       | baseline (lower bound)        | b2bgemm
    | load  | N * O + O * P + M * N + N * P | M / m * P / p * N / n * (m * n + O / o * (n * o + o * p))
    | store | N * P + M * P                 | M * P
    b2bgemm is always better on stores, but for loads we need to find out beneficial cases using this function
    """

b2b_gemm_configs: Incomplete

def is_b2b_gemm_good_on(is_left_assoc: bool, A_node: torch.fx.Node, B_node: torch.fx.Node, C_node: torch.fx.Node) -> bool:
    """
    checks whether the sizes are good for b2b_gemm
    """
def unoptimized_b2b_gemm(is_left_assoc: bool, subgraph: Subgraph, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """
    The unoptimized version is used as a fallback when the b2b_gemm kernel is not beneficial.
    """

unoptimized_choice: Incomplete

def build_subgraph_buffer(args: list[TensorBox], subgraph: Subgraph):
    """
    This function is adapted from ../kernel/flex_attention.py.
    The goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that are passed into the subgraph
        subgraph: The Subgraph ir for which to produce the output node
    """
def create_placeholder(name: str, dtype: torch.dtype, device: torch.device) -> TensorBox:
    """
    Creates a placeholder input buffers for producing subgraph_output
    """
def tuned_b2b_gemm(is_left_assoc: bool, subgraph: Subgraph, A: torch._inductor.ir.TensorBox, B: torch._inductor.ir.TensorBox, C: torch._inductor.ir.TensorBox, *, layout=None) -> torch._inductor.ir.TensorBox: ...
def b2b_gemm_handler(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node) -> None: ...
