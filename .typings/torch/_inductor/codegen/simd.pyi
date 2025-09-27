import contextlib
import dataclasses
import sympy
import torch
from .. import config as config, ir as ir, scheduler as scheduler
from ..._dynamo.utils import counters as counters
from ..analyze_preserves_zero_mask import prologue_preserves_zero_mask as prologue_preserves_zero_mask
from ..codecache import code_hash as code_hash
from ..dependencies import MemoryDep as MemoryDep, StarDep as StarDep, WeakDep as WeakDep
from ..ir import IRNode as IRNode
from ..optimize_indexing import indexing_dtype_strength_reduction as indexing_dtype_strength_reduction
from ..runtime.runtime_utils import green_text as green_text, yellow_text as yellow_text
from ..scheduler import BaseSchedulerNode as BaseSchedulerNode, BaseScheduling as BaseScheduling, WhyNoFuse as WhyNoFuse
from ..utils import IndentedBuffer as IndentedBuffer, Placeholder as Placeholder, cache_on_self as cache_on_self, expr_fits_within_32bit as expr_fits_within_32bit, get_dtype_size as get_dtype_size, prefix_is_reduction as prefix_is_reduction, set_kernel_post_grad_provenance_tracing as set_kernel_post_grad_provenance_tracing, sympy_index_symbol as sympy_index_symbol, sympy_product as sympy_product, sympy_subs as sympy_subs, unique as unique
from ..virtualized import OpsWrapper as OpsWrapper, V as V, ops as ops
from .block_analysis import BlockPatternMatcher as BlockPatternMatcher
from .common import CSEVariable as CSEVariable, Kernel as Kernel, PythonPrinter as PythonPrinter, index_prevent_reordering as index_prevent_reordering
from .multi_kernel import MultiKernel as MultiKernel
from .simd_kernel_features import DisableReduction as DisableReduction, EnableReduction as EnableReduction, NodeScheduleEntry as NodeScheduleEntry, NodeScheduleMarker as NodeScheduleMarker, SIMDKernelFeatures as SIMDKernelFeatures
from _typeshed import Incomplete
from collections.abc import Iterable, Iterator, Sequence
from torch._inductor.tiling_utils import CoalesceVarAnalysis as CoalesceVarAnalysis, analyze_memory_coalescing as analyze_memory_coalescing
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols as free_unbacked_symbols
from torch.fx.immutable_collections import immutable_dict as immutable_dict
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._sympy.functions import FloorDiv as FloorDiv, Identity as Identity, ModularIndexing as ModularIndexing
from torch.utils._sympy.symbol import SymT as SymT, free_symbol_is_type as free_symbol_is_type, prefix_str as prefix_str, symbol_is_type as symbol_is_type
from typing import Any, Callable, Generic, no_type_check
from typing_extensions import TypeVar

log: Incomplete
perf_hint_log: Incomplete
schedule_log: Incomplete
fusion_log: Incomplete
pexpr: Incomplete
all_prefixes: Incomplete

def get_max_tiles(default: int = 2) -> int: ...

@dataclasses.dataclass
class IterationRanges:
    """
    Each range tree represents multiple sets of iteration indexing
    in a single tiled dimension in the output kernel.

    If you have two loops ranges one (4, 3, 2) and another (4, 6),
    then the range tree will be:
            4 (i0)
        3 (i1)  6 (i3)
        2 (i2)
    Where i0 is shared between both loops, but then the split into
    different indexing vars.  All loop ranges must iterate over
    the same number of elements.
    """
    name = ...
    var_list = ...
    var_ranges = ...
    numel = ...
    prefix = ...
    divisor = ...
    length = ...
    kernel = ...
    root = ...
    def __init__(self, name: str, var_list: list[sympy.Symbol], var_ranges: dict[sympy.Symbol, sympy.Expr], numel: sympy.Expr, prefix: str, *, kernel: SIMDKernel, divisor=..., length=..., root: IterationRangesRoot) -> None: ...
    @property
    @cache_on_self
    @no_type_check
    def is_reduction(self): ...
    def symbol(self) -> sympy.Symbol: ...
    @property
    @cache_on_self
    @no_type_check
    def symt(self): ...

class IterationRangesRoot(IterationRanges):
    """
    Root of a iteration range tree that represents a single
    tiled dimension in the output kernel. It contains multiple
    sets of iteration represented with IterationRangesEntry.
    """
    index: Incomplete
    nodes: dict[sympy.Expr, IterationRangesEntry]
    pid_cache: dict[str, str]
    is_loop: Incomplete
    tensor_dim: Incomplete
    grid_dim: Incomplete
    has_zdim: Incomplete
    def __init__(self, name: str, numel: sympy.Expr, prefix: str, index: int, kernel: SIMDKernel, pid_cache: dict[str, str] | None = None, *, is_loop: bool, tensor_dim: int | None, grid_dim: int | None, has_zdim: bool) -> None: ...
    def __repr__(self) -> str: ...
    def cache_clear(self) -> None: ...
    def index_sym(self) -> sympy.Symbol: ...
    def lookup(self, divisor: sympy.Expr, length: sympy.Expr) -> IterationRangesEntry:
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
    def construct_entries(self, lengths: list[sympy.Expr]) -> list[IterationRangesEntry]: ...
    def construct(self, lengths: list[sympy.Expr]) -> list[sympy.Symbol]: ...
    def vars_and_sizes(self, index: sympy.Expr) -> tuple[list[sympy.Symbol], list[sympy.Expr]]:
        """Figure out vars from this tree used in index"""

class IterationRangesEntry(IterationRanges):
    parent: Incomplete
    codegen: Incomplete
    expr: Incomplete
    def __init__(self, name: str, divisor: sympy.Expr, length: sympy.Expr, expr: sympy.Expr, parent: IterationRanges) -> None: ...
    def __repr__(self) -> str: ...
    name: Incomplete
    def set_name(self, name: str) -> None: ...
    def cache_clear(self) -> None: ...
    def _codegen(self) -> str: ...
    def precomputed_args(self) -> list[sympy.Expr]: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

def constant_repr(value: int | float) -> str: ...
CSEVariableType = TypeVar('CSEVariableType', bound=CSEVariable, default=CSEVariable)

class SIMDKernel(Kernel[CSEVariableType], Generic[CSEVariableType]):
    """
    Common base class for Triton/Halide codegen which both use flattened indexing rather than loop nests.
    """
    sexpr: Callable[[sympy.Expr], str]
    kexpr: Callable[[sympy.Expr], str]
    allow_block_ptr: bool
    kernel_name: str
    features: Incomplete
    mutations: Incomplete
    body: Incomplete
    indexing_code: Incomplete
    numels: Incomplete
    range_trees: list[IterationRangesRoot]
    range_tree_nodes: dict[sympy.Symbol, IterationRangesEntry]
    iter_vars_count: Incomplete
    inside_reduction: Incomplete
    cooperative_reduction: bool
    tiling_scores: dict[str, sympy.Expr] | None
    persistent_reduction: bool
    no_x_dim: Incomplete
    code_hash: str | None
    simplify_indexing: Incomplete
    def __init__(self, tiling: dict[str, sympy.Expr], features: SIMDKernelFeatures, pid_cache: dict[str, str] | None = None, override_persistent_reduction: bool | None = None, override_cooperative_reduction: bool | None = None, tiling_scores: dict[str, sympy.Expr] | None = None) -> None: ...
    @property
    @cache_on_self
    @no_type_check
    def num_reduction_dims(self): ...
    def dtype_to_str(self, dtype: torch.dtype) -> str: ...
    def get_index_dtype_as_torch_dtype(self) -> torch.dtype: ...
    @property
    def index_dtype(self) -> str: ...
    def want_no_x_dim(self) -> bool: ...
    def construct_range_trees(self, pid_cache: dict[str, str] | None, inside_reduction: bool, is_reduction: bool, numels: dict[str, sympy.Expr], no_x_dim: bool) -> list[IterationRangesRoot]: ...
    def initialize_range_tree(self, pid_cache: dict[str, str]) -> None: ...
    def finalize_indexing(self, indices: Sequence[sympy.Expr]) -> None:
        """
        Hook called right before codegen with every index that will be
        used in the fused kernel.
        """
    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None: ...
    def should_use_cooperative_reduction(self) -> bool: ...
    def should_use_persistent_reduction(self) -> bool: ...
    def var_ranges(self) -> dict[sympy.Symbol, sympy.Expr]: ...
    def triton_tensor_ndim(self) -> int: ...
    def indexing_size_str(self, i: int) -> str: ...
    def dense_size_list(self) -> list[str]: ...
    def dense_size_str(self) -> str: ...
    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr: ...
    def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot) -> sympy.Expr: ...
    def _combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot) -> sympy.Expr:
        """
        More aggressive simplification to merge contiguous dims
        """
    def disable_reduction(self) -> contextlib.AbstractContextManager[None]: ...
    def set_ranges(self, *lengths: sympy.Expr) -> list[sympy.Symbol]: ...
    @staticmethod
    def _split_iteration_ranges(groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]) -> tuple[list[list[sympy.Expr]], list[list[Callable[[list[sympy.Expr]], sympy.Expr]]]]: ...
    @classmethod
    def prepare_split_iteration_lengths(cls, groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]], reduction_numel: sympy.Expr = ...) -> Sequence[Sequence[sympy.Expr]]:
        """Fill in the reduction numel of lengths if missing"""
    @classmethod
    def is_compatible(cls, groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]], reduction_numel: sympy.Expr = ...) -> bool: ...
    def split_and_set_ranges(self, lengths: Sequence[Sequence[sympy.Expr]]) -> list[list[sympy.Expr]]: ...
    @classmethod
    def map_kernel_groups_to_node_sizes(cls, groups: Sequence[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]], set_ranges) -> list[list[sympy.Expr]]:
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
    def is_indirect_indexing(self, index: sympy.Expr) -> bool: ...
    def is_broadcasted(self, index: sympy.Expr) -> bool: ...
    def index_to_str(self, index: sympy.Expr) -> str:
        '''
        Convert an index expr to a string that can be used in output code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the generated kernel.

        Index expressions often need to be passed in as arguments to the triton kernel.
        Rename_indexing and codegen_indexing keep track of the needed indices and add
        new parameters to the function signature.
        '''
    def prepare_indexing(self, index: sympy.Expr) -> sympy.Expr: ...
    def active_range_trees(self) -> list[IterationRangesRoot]: ...
    def codegen_indexing(self, expr: sympy.Expr) -> sympy.Expr: ...
    def codegen_nan_check(self) -> None: ...
    def call_kernel(self, name: str, node: IRNode | None = None) -> None: ...
    _load_mask: Incomplete
    _load_other: Incomplete
    @contextlib.contextmanager
    def mask_loads(self, mask: str | OpsWrapper, value: int | float) -> Iterator[str]:
        """Context manager to add an additional mask to tl.load/store"""
    def get_strides_of_load(self, index: sympy.Expr) -> dict[sympy.Symbol, sympy.Expr]:
        """
        This gets the stride of the index for each of the tiling variables
        (technically, it does it at index 0)

        For example, if
        xindex = x0 + 512*x1 + 1024*r0
        x0 = (xindex//512)
        x1 = (xindex % 512)
        r0 = rindex // 1024

        this function would return
        {xindex: 512, rindex: 1024}
        """
    @staticmethod
    def _map_tuple_or_scalar(fn, value): ...
    def estimate_kernel_num_bytes(self):
        '''
        Try the best to estimate the total size (in bytes) of the
        kernel\'s inputs and outputs, which is used for estimating the memory
        throughput of this kernel. This information is used for checking how
        far we are from the peak memory bandwidth. It\'s important that
        we want to avoid overestimating the sizes of the inputs and outputs,
        because it can wrongfully give us a very large memory traffic value,
        which may be even larger than the theoretical bandwidth and thus
        become very misleading. This is particularly problematic for cases
        where we slice some inputs. In those cases, we should only count
        the size of the "slices" instead of the original inputs, because
        only the slices contribute to the real memory traffic.
        '''
    def warn_mix_layout(self, kernel_name) -> None:
        """
        Print message if the kernel have mixed layout inputs.
        Only care about 4D tensor for now.
        """
    def welford_reduce_fallback(self, dtype, value): ...
    def prepare_softmax_twopass_fallback(self, dtype, value): ...
    def codegen_kernel(self) -> None: ...
    def codegen_body(self) -> None: ...
    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry): ...

class SIMDScheduling(BaseScheduling):
    """
    Single Instruction Multiple Data parent class used for fusion across
    multiple different backends.
    """
    kernel_type: type[Any]
    def group_fn(self, sizes): ...
    def can_fuse(self, node1, node2):
        """
        Hook called by Scheduler to determine if the Triton backend
        can fuse node1 and node2.  These nodes might already be
        FusedSchedulerNodes.
        """
    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse
    def generate_node_schedule(self, nodes, numel, rnumel): ...
    def codegen_node(self, node: scheduler.FusedSchedulerNode | scheduler.SchedulerNode):
        """
        Given a set of pre-fused nodes, generate a Triton kernel.
        """
    @staticmethod
    def can_use_32bit_indexing(numel: sympy.Expr, buffers: Iterable[ir.Buffer | ir.TensorBox | ir.TorchBindObject | ir.IRNode]) -> bool: ...
    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures): ...
    def create_kernel_choices(self, kernel_features: SIMDKernelFeatures, kernel_args, kernel_kwargs) -> list[SIMDKernel]: ...
    def codegen_node_schedule_with_kernel(self, node_schedule, kernel) -> None: ...
    def codegen_template(self, template_node, epilogue_nodes, prologue_nodes, *, only_gen_src_code: bool = False) -> str | None:
        """
        Codegen a triton template

        If `only_gen_src_code` the src code will be returned instead of codegen'd into the wrapper
        """
    def codegen_sync(self) -> None: ...
    def generate_combo_kernel_code(self, subkernel_nodes: list[BaseSchedulerNode], custom_part_algorithm: bool, enable_autotune: bool, mixed_sizes: bool, only_gen_src_code: bool = False) -> list[tuple[str, Any, Any]]: ...
    def codegen_combo_kernel(self, combo_kernel_node) -> None: ...
    @classmethod
    def candidate_tilings(cls, node, numel, reduction_numel) -> list[CandidateTiling]: ...
    @classmethod
    def create_tiling(cls, pw_tiling: Sequence[sympy.Expr], reduction_tiling: Sequence[sympy.Expr]) -> dict[str, sympy.Expr]:
        """
        Create a tiling dict from pointwise and reduction splits.
        """
    @classmethod
    def create_partial_tiling(cls, tiling: Sequence[sympy.Expr], is_pointwise: bool) -> dict[str, sympy.Expr]: ...
    @classmethod
    def complete_partial_tiling(cls, tiling: dict[str, sympy.Expr], numel: sympy.Expr, reduction_numel: sympy.Expr) -> dict[str, sympy.Expr]:
        """
        Given a tiling for only pointwise or reduction dimensions, adds the missing one.
        """
    @classmethod
    def get_nd_tilings(cls, node_schedule, pointwise_numel, reduction_numel) -> list[dict[str, tuple[sympy.Expr]]]:
        """
        Creates N-dimensional tiling candidates, attempting to simplify loads/stores
        by tiling the kernel into higher dimensions.

        Returns a list of tilings ranked by dimensionality.
        """
    @classmethod
    def compute_tiling_strategy(cls, node_schedule: list[NodeScheduleEntry], pointwise_numel: sympy.Expr, reduction_numel: sympy.Expr, coalesce_analysis: CoalesceVarAnalysis) -> tuple[dict[str, sympy.Expr], dict[str, sympy.Expr] | None]:
        """
        Generates a tiling, and a score of each tile according to each tile's coalesced memory accesses.
        """
    @classmethod
    def tiling_is_compatible(cls, node_schedule: list[NodeScheduleEntry], numel: sympy.Expr, reduction_numel: sympy.Expr, tiling: dict[str, sympy.Expr]): ...
    @classmethod
    def get_first_compatible_tiling(cls, node_schedule: list[NodeScheduleEntry], numel: sympy.Expr, reduction_numel: sympy.Expr, ranked_tilings: list[dict[str, sympy.Expr]]): ...
    @classmethod
    def select_tiling(cls, node_schedule, numel, reduction_numel=..., coalesce_analysis: CoalesceVarAnalysis | None = None) -> dict[str, sympy.Expr]: ...
    @classmethod
    def get_tiling_and_scores(cls, node_schedule, numel, reduction_numel=..., coalesce_analysis: CoalesceVarAnalysis | None = None) -> tuple[dict[str, sympy.Expr], dict[str, sympy.Expr] | None]:
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
    def flush(self) -> None: ...
    def ready_to_flush(self) -> bool: ...
    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel: bool = False): ...
    def codegen_comment(self, node_schedule) -> None: ...
    def define_kernel(self, src_code, node_schedule, kernel) -> None: ...

@dataclasses.dataclass(frozen=True)
class CandidateTiling:
    tiling: dict[str, sympy.Expr]
    score: int
    name: str | None = ...
    @staticmethod
    def is_good_size(s):
        """Somewhat arbitrary heuristic used to boost scores for some sizes"""

class CantSplit(Exception): ...
