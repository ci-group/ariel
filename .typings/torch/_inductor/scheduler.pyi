import dataclasses
import sympy
import torch
from . import comms as comms, config as config, dependencies as dependencies, ir as ir, metrics as metrics
from .analyze_preserves_zero_mask import can_codegen_without_upcasts as can_codegen_without_upcasts
from .codegen.common import BackendFeature as BackendFeature, Kernel as Kernel, get_scheduling_for_device as get_scheduling_for_device
from .comm_analysis import estimate_nccl_collective_runtime as estimate_nccl_collective_runtime
from .dependencies import Dep as Dep, MemoryDep as MemoryDep, StarDep as StarDep, WeakDep as WeakDep
from .exc import GPUTooOldForTriton as GPUTooOldForTriton, TritonMissing as TritonMissing
from .fx_utils import count_flops_fx as count_flops_fx, countable_fx as countable_fx
from .ir import GraphPartitionSignature as GraphPartitionSignature, MultiOutput as MultiOutput, MultiOutputLayout as MultiOutputLayout, NoneLayout as NoneLayout, get_device_type as get_device_type
from .loop_body import LoopBody as LoopBody
from .memory import MemoryPlanningInfoForBuffer as MemoryPlanningInfoForBuffer, MemoryPlanningInfoForNode as MemoryPlanningInfoForNode
from .runtime.runtime_utils import green_text as green_text, red_text as red_text
from .sizevars import SimplifyIndexing as SimplifyIndexing
from .utils import GraphPartitionMap as GraphPartitionMap, IndentedBuffer as IndentedBuffer, cache_on_self as cache_on_self, cmp as cmp, device_need_guard as device_need_guard, get_device_tflops as get_device_tflops, get_dtype_size as get_dtype_size, get_gpu_dram_gbps as get_gpu_dram_gbps, is_collective as is_collective, is_cudagraph_unsafe_op as is_cudagraph_unsafe_op, is_gpu as is_gpu, is_multi_outputs_template as is_multi_outputs_template, is_output_of_multi_outputs_template as is_output_of_multi_outputs_template, is_wait as is_wait, sympy_product as sympy_product
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._dynamo.utils import counters as counters, dynamo_timed as dynamo_timed
from torch._inductor.codecache import LambdaFuture as LambdaFuture, PyCodeCache as PyCodeCache
from torch._inductor.metrics import get_metric_table as get_metric_table, is_metric_table_enabled as is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_symbols as free_symbols
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._sympy.symbol import SymT as SymT, free_symbol_is_type as free_symbol_is_type, symbol_is_type as symbol_is_type
from torch.utils._triton import has_triton as has_triton
from types import ModuleType
from typing import Any, Callable

log: Incomplete
fusion_log: Incomplete
loop_ordering_log: Incomplete
PartitionType: Incomplete

@dataclasses.dataclass
class SchedulerBuffer:
    scheduler: Scheduler
    node: ir.Buffer
    defining_op: BaseSchedulerNode | None
    users: list[NodeUser] = dataclasses.field(default_factory=list)
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(default_factory=MemoryPlanningInfoForBuffer)
    def defining_op_name(self) -> str: ...
    def __hash__(self) -> int: ...
    def debug_str(self) -> str: ...
    def get_name(self) -> str: ...
    def allocate(self) -> None: ...
    def can_free(self) -> bool: ...
    def set_users(self, users: list[NodeUser]) -> None: ...
    def get_aliases(self) -> Sequence[str]: ...
    def get_mutations(self) -> Sequence[str]: ...
    def get_device(self) -> torch.device | None: ...

@dataclasses.dataclass
class SchedulerDonatedBuffer(SchedulerBuffer):
    defining_op: BaseSchedulerNode | None = ...

class BaseSchedulerNode:
    group: tuple[torch.device, tuple[tuple[sympy.Expr, ...], ...]]
    read_writes: dependencies.ReadWrites
    unmet_dependencies: OrderedSet[Dep]
    min_order: int
    max_order: int
    mpi_node: MemoryPlanningInfoForNode
    scheduler: Scheduler
    debug_device_str: Callable[[BaseSchedulerNode], list[str]]
    def __init__(self, scheduler: Scheduler) -> None: ...
    node: ir.Operation | None
    ancestors: OrderedSet[str]
    last_usage: Incomplete
    written: bool
    outputs: list[SchedulerBuffer]
    outputs_by_name: dict[str, SchedulerBuffer]
    def _init_from_node(self, node: ir.Operation) -> None: ...
    def __repr__(self) -> str: ...
    def debug_str(self) -> str:
        """Longer form printout for trace logs"""
    def debug_str_extra(self) -> str: ...
    def _debug_str_for_device(self) -> list[str]: ...
    def debug_str_short(self) -> str: ...
    def log_details(self) -> None: ...
    def reorder_loops_by_dep_pair(self, self_dep: MemoryDep, other_dep: MemoryDep) -> None: ...
    def update_mutated_names(self, renames: dict[str, str]) -> None: ...
    def add_fake_dep(self, dep: Dep) -> None: ...
    def has_aliasing_or_mutation(self) -> bool: ...
    def set_read_writes(self, rw: dependencies.ReadWrites) -> None: ...
    def set_last_usage(self, future_used_buffers: OrderedSet[str], mutation_real_name: dict[str, str]) -> None: ...
    def mark_run(self) -> None: ...
    def used_buffer_names(self) -> OrderedSet[str]: ...
    def used_or_aliased_buffer_names(self) -> OrderedSet[str]: ...
    def prune_deps(self) -> None: ...
    def prune_weak_deps(self) -> None: ...
    def prune_redundant_deps(self, name_to_fused_node: dict[str, BaseSchedulerNode]) -> None: ...
    def get_name(self) -> str: ...
    def get_first_name(self) -> str: ...
    @cache_on_self
    def get_operation_names(self) -> OrderedSet[str]: ...
    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]: ...
    @cache_on_self
    def can_codegen_in_low_precision(self) -> bool: ...
    @cache_on_self
    def can_codegen_without_upcasts(self) -> bool: ...
    def get_nodes(self) -> Sequence[BaseSchedulerNode]: ...
    def get_outputs(self) -> Sequence[SchedulerBuffer]: ...
    def get_output(self, buf_name: str) -> SchedulerBuffer: ...
    def get_device(self) -> torch.device | None: ...
    def is_cpu(self) -> bool: ...
    def is_gpu(self) -> bool: ...
    def is_reduction(self) -> bool: ...
    def is_split_scan(self) -> bool: ...
    def is_template(self) -> bool: ...
    def is_extern(self) -> bool: ...
    def is_foreach(self) -> bool: ...
    def can_inplace(self, read_dep: dependencies.Dep) -> bool: ...
    def has_side_effects(self) -> bool: ...
    def decide_inplace_update(self) -> None:
        """
        Decide if there should be inplace updates for the node
        and record the decision in the active kernel.
        """
    def codegen_originating_info(self, buffer: IndentedBuffer, only_once: bool = True) -> None: ...
    @cache_on_self
    def get_read_write_buffers_sizes(self) -> int: ...
    @cache_on_self
    def get_read_buffer_sizes(self) -> int: ...
    @cache_on_self
    def get_write_buffer_sizes(self) -> int: ...
    def get_read_write_buffers_sizes_impl(self, include_reads: bool, include_writes: bool) -> int: ...
    def get_read_write_buffer_accesses(self, include_reads: bool, include_writes: bool) -> dict[str, int]:
        """
        Counting the number of bytes accessed for a kernel is
        surprisingly tricky. In particular, there is a differentiation
        between 'theoretical' memory accesses and practical memory
        accesses. For example, a layernorm kernel may actually access an
        input 3 times, but in theory, it only needs to access its input
        once (and may be optimized to do so through say, persistent
        reductions)

        Another example is that even though a buffer is passed in, we may
        not access the entire buffer. This may occur if we are accessing
        a slice of the buffer. Another tricky case is for indirect
        indexing, where the amount of bytes accessed depends on the
        values of the input.

        What this function aims to compute is the memory accesses for
        worst-case inputs, best-case optimization. What this means is
        that for each buffer we compute the amount of potential accesses in two ways and take the minimum.

        1. Numel in ranges multiplied by number of deps the buffer has
        2. The buffer size

        Returns memory accesses per buffer.
        """
    @cache_on_self
    def estimate_flops(self) -> int | None: ...
    @cache_on_self
    def get_estimated_runtime(self) -> float:
        """
        Returns estimated op runtime in nanoseconds (ns)
        """
    def get_template_node(self) -> ir.TemplateBuffer | None: ...
    def get_template_node_or_throw(self) -> ir.TemplateBuffer: ...
    @staticmethod
    def get_prologue_template_epilogue(nodes: list[BaseSchedulerNode]) -> tuple[list[BaseSchedulerNode], BaseSchedulerNode, list[BaseSchedulerNode]]:
        """
        For the list of nodes, get the prologue, template, and epilogue
        """

class WhyNoFuse:
    __slots__: Incomplete
    reason: str
    args: tuple[Any, ...]
    name1: Incomplete
    name2: Incomplete
    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> None: ...
    def __call__(self, reason: str, *args: Any) -> None: ...
    def __str__(self) -> str: ...

def pformat(obj: Any) -> str: ...

class OutputNode:
    unmet_dependencies: Incomplete
    def __init__(self, dep: StarDep) -> None: ...
    def is_reduction(self) -> bool: ...
    def get_inputs_that_alias_output(self) -> Sequence[str]: ...
    def get_name(self) -> str: ...
    __repr__ = get_name

def _prune_redundant_deps(node: BaseSchedulerNode, name_to_fused_node: dict[str, BaseSchedulerNode], name_to_buf: dict[str, SchedulerBuffer]) -> None:
    """
    Prunes weakdeps intended for mutation ordering
    on an upstream fused node if after fusion there is another dependency
    on the fused upstream node, making the weakdep redundant

    In essence this enforces an ordering on fusions. As fusions occur, weakdeps will
    be incrementally removed, enabling other fusions, ensuring they are fused in order.
    """

class ExternKernelSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: Scheduler, node: ir.Operation) -> None: ...
    def debug_str_extra(self) -> str: ...
    def is_extern(self) -> bool: ...
    def has_side_effects(self) -> bool: ...

class NopKernelSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: Scheduler, node: ir.Operation) -> None: ...

class SchedulerNode(BaseSchedulerNode):
    _sizes: tuple[Sequence[sympy.Expr], ...]
    _body: LoopBody
    def __init__(self, scheduler: Scheduler, node: ir.ComputedBuffer | ir.TemplateBuffer) -> None: ...
    group: Incomplete
    def _compute_attrs(self, extra_indexing_constraints: tuple[dict[Any, Any], list[Any]] | None = None, recompute_sizes_body_func: Callable[..., Any] | None = None) -> None: ...
    def recompute_size_and_body(self, extra_indexing_constraints: tuple[dict[Any, Any], list[Any]] | None = None, recompute_sizes_body_func: Callable[..., Any] | None = None) -> None: ...
    def refresh_dependencies(self, normalize: bool, need_clear_tiling_cache: bool) -> None: ...
    def apply_new_loop_order(self, new_order: Sequence[int]) -> None: ...
    def merge_loops(self) -> None: ...
    def reorder_loops_by_dep_pair(self, self_dep: MemoryDep, other_dep: MemoryDep) -> None: ...
    def debug_str_extra(self) -> str: ...
    def get_ranges(self) -> Sequence[Sequence[sympy.Expr]]: ...
    def is_reduction(self) -> bool: ...
    def is_split_scan(self) -> bool: ...
    def is_template(self) -> bool: ...
    def get_template_node(self) -> ir.TemplateBuffer | None: ...
    def run(self, *index_vars: Sequence[sympy.Expr]) -> None: ...
    def ranges_from_index_vars(self, index_vars: Sequence[Sequence[sympy.Expr]]) -> dict[sympy.Expr, sympy.Expr]: ...
    def codegen(self, index_vars: Sequence[Sequence[sympy.Expr]]) -> None: ...
    def pointwise_or_reduction_read_writes(self, pointwise: bool = True) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in either the pointwise or the reduction axes.
        """
    @cache_on_self
    def pointwise_read_writes(self) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in the non-reduction axes.
        """
    @cache_on_self
    def reduction_read_writes(self) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in the reduction axes.
        """
    def can_inplace(self, read_dep: dependencies.Dep) -> bool: ...
    @cache_on_self
    def _get_atomic_add_buffers(self) -> OrderedSet[str]: ...

def refresh_group_node_dependencies(group_snode: FusedSchedulerNode | GroupedSchedulerNode) -> None: ...
def init_group_node(group_snode: FusedSchedulerNode | GroupedSchedulerNode, scheduler: Scheduler, snodes: list[BaseSchedulerNode]) -> None: ...

class FusedSchedulerNode(BaseSchedulerNode):
    '''
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    '''
    snodes: list[BaseSchedulerNode]
    @classmethod
    def fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> FusedSchedulerNode: ...
    @cache_on_self
    def estimate_flops(self) -> int | None: ...
    def reorder_loops_by_dep_pair(self, self_dep: MemoryDep, other_dep: MemoryDep) -> None: ...
    users: list[NodeUser]
    group: Incomplete
    def __init__(self, scheduler: Scheduler, snodes: list[BaseSchedulerNode]) -> None: ...
    @cache_on_self
    def get_name(self) -> str: ...
    def get_first_name(self) -> str: ...
    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]: ...
    def get_outputs(self) -> list[SchedulerBuffer]: ...
    def debug_str_extra(self) -> str: ...
    def debug_str_short(self) -> str: ...
    def set_last_usage(self, future_used_buffers: OrderedSet[str], mutation_real_name: dict[str, str]) -> None: ...
    @cache_on_self
    def used_buffer_names(self) -> OrderedSet[str]: ...
    @cache_on_self
    def used_or_aliased_buffer_names(self) -> OrderedSet[str]: ...
    def get_nodes(self) -> Sequence[BaseSchedulerNode]: ...
    def __repr__(self) -> str: ...
    @cache_on_self
    def is_reduction(self) -> bool: ...
    @cache_on_self
    def is_split_scan(self) -> bool: ...
    @cache_on_self
    def is_template(self) -> bool: ...
    @cache_on_self
    def get_template_node(self) -> ir.TemplateBuffer | None: ...
    def get_device(self) -> torch.device: ...
    @cache_on_self
    def has_aliasing_or_mutation(self) -> bool: ...
    def update_mutated_names(self, renames: dict[str, str]) -> None: ...
    def add_fake_dep(self, name: Dep) -> None: ...
    def can_inplace(self, read_dep: dependencies.Dep) -> bool: ...
    def debug_str(self) -> str:
        """Longer form printout for trace logs"""

class ForeachKernelSchedulerNode(FusedSchedulerNode):
    """
    This is a schedular node that consists of a set of scheduler nodes that
    has no data dependencies among them and can be executed in parallel.
    """
    def get_consumer_subnode_for(self, producer: BaseSchedulerNode) -> BaseSchedulerNode | None: ...
    def get_producer_subnode_for(self, consumer: BaseSchedulerNode) -> BaseSchedulerNode | None: ...
    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool: ...
    @classmethod
    def fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> ForeachKernelSchedulerNode: ...
    read_to_node: Incomplete
    name_to_node: Incomplete
    scheduler: Incomplete
    snodes: Incomplete
    node: Incomplete
    users: list[NodeUser]
    unmet_dependencies: Incomplete
    min_order: Incomplete
    max_order: Incomplete
    ancestors: Incomplete
    outputs_by_name: dict[str, SchedulerBuffer]
    use_custom_partition_algo: Incomplete
    group: Incomplete
    origins: Incomplete
    enable_autotune: Incomplete
    def __init__(self, scheduler: Scheduler, snodes: list[BaseSchedulerNode], use_custom_partition_algo: bool, prev_node_1: BaseSchedulerNode | None = None, prev_node_2: BaseSchedulerNode | None = None, enable_autotune: bool = False) -> None: ...
    @classmethod
    def combinable_nodes(cls, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]: ...
    @staticmethod
    def _default_group_nodes_for_combo_kernels(scheduler: Scheduler) -> list[list[BaseSchedulerNode]]:
        """
        Returns a list of lists of nodes that are to be grouped together.
        """
    group_algorithm_for_combo_kernels: Callable[[Scheduler], list[list[BaseSchedulerNode]]]
    @staticmethod
    def set_group_algorithm_for_combo_kernels(custom_group_algorithm: Callable[[Scheduler], list[list[BaseSchedulerNode]]]) -> None: ...
    @staticmethod
    def group_nodes_for_combo_kernels(scheduler: Scheduler) -> list[list[BaseSchedulerNode]]: ...
    def mark_run(self) -> None: ...
    def codegen(self) -> None: ...
    def is_foreach(self) -> bool: ...
    def get_subkernel_nodes(self) -> list[BaseSchedulerNode]:
        """Returns a list of nodes which comprise the combo kernel.
        These nodes may be vertically fused."""
    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        """Returns all nodes contained in this kernel, unpacking fused nodes
        into their constituent scheduler nodes."""
    def get_first_name(self) -> str: ...
    def prune_redundant_deps(self, name_to_fused_node: dict[str, BaseSchedulerNode]) -> None: ...

class GroupedSchedulerNode(BaseSchedulerNode):
    '''
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be *grouped* together (it does not allow another node to be scheduled
    in between its constituent nodes, nor does it allow another node to fuse into any of its constituent nodes).
    The way it does this is by maintaining its unmet dependencies as the union of its constituent nodes.
    Fusion will still happen among the nodes within each GroupedSchedulerNode.
    At codegen time, this scheduler node will be unpacked and codegen is called on each constituent node.
    '''
    snodes: list[BaseSchedulerNode]
    @classmethod
    def create(cls, snodes: list[BaseSchedulerNode]) -> GroupedSchedulerNode: ...
    def __init__(self, scheduler: Scheduler, snodes: list[BaseSchedulerNode]) -> None: ...
    def unpack(self) -> list[BaseSchedulerNode]:
        """
        Do fusion among nodes within this GroupedSchedulerNode,
        and then unpack this GroupedSchedulerNode into regular nodes.
        """
    def add_fake_dep(self, fake_dep: Dep) -> None: ...
    @cache_on_self
    def get_name(self) -> str: ...
    def get_first_name(self) -> str: ...
    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]: ...
    def get_outputs(self) -> list[SchedulerBuffer]: ...
    @cache_on_self
    def estimate_flops(self) -> int | None: ...
    def get_nodes(self) -> Sequence[BaseSchedulerNode]: ...
    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool: ...

def pick_loop_order(stride_lengths: list[list[int]], sizes: Sequence[sympy.Expr], priority_idx: tuple[int, ...] = ()) -> list[int]:
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

@dataclasses.dataclass
class NodeUser:
    node: BaseSchedulerNode | OutputNode
    can_inplace: bool = ...
    is_weak: bool = ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def get_name(self) -> str: ...
    def merge(self, other: NodeUser) -> NodeUser: ...

_post_grad_graph_counter: Incomplete

class Scheduler:
    """
    A Scheduler is a graph of BaseSchedulerNodes. It is responsible for
    optimizations such as fusion, reorder, and graph partition.
    """
    __dep_size_hint_cache: dict[Dep, int]
    def __init__(self, nodes: list[ir.Operation]) -> None: ...
    backends: dict[torch.device, BaseScheduling]
    post_grad_graph_id: Incomplete
    _graph_partition_counter: Incomplete
    completed_operations: OrderedSet[str]
    available_buffer_names: Incomplete
    nodes: Incomplete
    name_to_donated_buffer: dict[str, SchedulerDonatedBuffer]
    name_to_node: dict[str, BaseSchedulerNode]
    name_to_buf: dict[str, SchedulerBuffer]
    name_to_fused_node: dict[str, BaseSchedulerNode]
    mutation_real_name: dict[str, str]
    mutation_renames: dict[str, str]
    num_orig_nodes: Incomplete
    logged_slow_fusion: Incomplete
    buffer_names_to_free: OrderedSet[str]
    origin_to_index: dict[torch.fx.Node, int]
    def _init(self, nodes: list[ir.Operation]) -> None: ...
    def get_donated_buffers(self) -> dict[str, SchedulerDonatedBuffer]: ...
    @property
    def current_device(self) -> torch.device | None: ...
    @current_device.setter
    def current_device(self, device: torch.device | None) -> None: ...
    def debug_draw_graph(self) -> None:
        """Generate an image of the graph for debugging"""
    def debug_print_nodes(self, label: str) -> None: ...
    def create_scheduler_node(self, node: ir.Operation) -> BaseSchedulerNode: ...
    def create_foreach_nodes(self) -> None: ...
    items: Incomplete
    membership: Incomplete
    def compute_dependencies(self) -> None:
        """
        Create dependency edges between nodes, handling aliasing and
        mutation properly.
        """
    def dead_node_elimination(self) -> None:
        """
        Remove any nodes without users
        """
    def topological_sort_schedule(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Ensure nodes is in topologically sorted order
        """
    def _get_unmet_dep_nodes(self, snode: BaseSchedulerNode) -> list[BaseSchedulerNode]: ...
    def _topological_sort_nodes(self) -> list[list[BaseSchedulerNode]]:
        """
        Sort nodes by their topological order, return a list of node lists.
        """
    def compute_ancestors(self) -> None:
        """
        Populate each node.ancestors
        """
    def merge_loops(self) -> None: ...
    def fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Combine eligible nodes into FusedSchedulerNodes.
        """
    def process_grouped_nodes(self) -> None:
        """
        Unpack GroupedSchedulerNode into regular nodes.
        """
    def benchmark_fused_nodes(self, nodes: Sequence[BaseSchedulerNode]) -> tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
    def generate_kernel_code_from_nodes(self, nodes: Sequence[BaseSchedulerNode], benchmark_kernel: bool) -> str:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
    def benchmark_codegened_module(self, module: ModuleType, device: torch.device) -> tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
    def finalize_multi_template_buffers(self) -> None:
        """
        Finalize a backing choice for MultiTemplateBuffers which did not already have a
        choice finalized through fusion. In the case of an extern choice, this will result
        in replacing the SchedulerNode.

        If a MultiTemplateBuffer did not have any fusion opportunities, finalizing a choice
        will force completion of compilation and benchmarking.
        """
    def _any_atomic_add(self, node_list: Sequence[BaseSchedulerNode]) -> bool: ...
    def speedup_by_fusion(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool | Callable[[], bool]:
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """
    def get_fused_node(self, node: BaseSchedulerNode) -> BaseSchedulerNode:
        """Look up the node in Scheduler name_to_fused_node"""
    def fuse_nodes_once(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Combine eligible nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuse(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
    def create_combo_kernel_nodes(self, num_ck_nodes: int | None = None) -> None:
        """
        Groups parallel nodes
        """
    def prune_redundant_deps(self, nodes: list[BaseSchedulerNode]) -> None: ...
    def get_possible_fusions(self, nodes: list[BaseSchedulerNode]) -> list[tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
    def will_fusion_create_cycle(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Finds whether there's a path from node1 to node2 (or vice-versa)
        caused indirectly by other fusions.
        """
    def can_fusion_increase_peak_memory(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Return true if fusing the two nodes can potentially increasing peak memory.

        The implementation is more like a heuristic since we don't really know if we are at peak
        or not when trying to fuse these two nodes. The order of nodes may change later which makes the
        peak memory estimation hard.

        Here is how we decide the LOWER BOUND of extra memory allocation if we fuse these 2 nodes:
        1. find all buffers read by each node with a single user. These buffers are supposed to
           be reused if we don't fuses these 2 nodes
        2. find the intersection of these buffers for the two node and sum the total buffer size.
           If we don't fuse these two nodes, we can at lease avoid this much memory allocation.
           Note that the extra memory allocation is not necessarily causing peak memory increase.
           This is just a heuristic.

        We return true only if the saving for fusion can not trade off the extra memory allocation.
        """
    def are_long_distant_nodes(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        This function prevents fusion for nodes that can increase memory
        footprint. This problem is more common in horizontal fusion, where nodes
        that are far apart in the original order get fused, lengthening the live
        intervals of tensors. This is very evident in models with activation
        checkpointing, where the recomputed nodes from different checkpointed
        regions get fused and significantly increase the memory footprint.

        The current attempt is a quick, possibly hacky, heuristic to prevent the
        fusion of nodes that are far away in the original order.

        A better but difficult to implement heurisitic would be to use live
        intervals of the buffers, find region of peak pressure in the original
        program and prevent fusion that crosses that peak region. We might need
        special care or good approximation in this implementation, as fusion of
        node changes live intervals, and re-computing live intervals and peak
        memory after each fusion can introduce large compilation overhead.
        """
    def decide_fusion_fail_reason(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode, common_buf_names: tuple[str] | OrderedSet[str]) -> str:
        """
        Try to decide reasons why fusion fail due to no shared memory even though
        there are common buffers.
        """
    def shared_data_after_reordering_loop(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> int:
        """
        Right now just greedily reorder the loop of node1 to be compatible with node2,
        but ideally we should have some heuristics to reorder the loop for node2
        to be compatible with node1 if that's more efficient.
        """
    def unfusable_node(self, node: BaseSchedulerNode) -> bool:
        """
        Is this node unfusable under any conditions.
        """
    def check_prologue_fusion_heuristics_fusable(self, prologue_node: BaseSchedulerNode, template_node: BaseSchedulerNode, why: WhyNoFuse) -> bool:
        """
        Heuristics to avoid benchmarking predictably slow prologue fusions
        """
    def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Determine if it is possible to combine node1 and node2 into a
        single fused node.
        """
    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Check if it is legal to fuse a consumer (node2) into a producer (node1).

        We can fuse them if all the reads of node2 either match
        corresponding writes in node1, or are written by nodes that can
        be scheduled before the fusion of node1 and node2.
        """
    def fusable_weak_dep(self, weak_dep: WeakDep, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool: ...
    def fusable_read_and_write(self, read: Dep, write: MemoryDep) -> bool: ...
    def dep_size_hint(self, dep: Dep) -> int: ...
    def score_fusion_memory(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> int:
        """
        The first term in our fusion score that estimates number of saved
        memory operations.
        """
    def get_possible_fusions_with_highest_priority(self, possible_fusions: list[tuple[BaseSchedulerNode, BaseSchedulerNode]]) -> list[tuple[BaseSchedulerNode, BaseSchedulerNode]]: ...
    def score_fusion_key(self, nodes: tuple[BaseSchedulerNode, BaseSchedulerNode]) -> Any:
        """
        Shim for list.sort(key=...)
        """
    def compute_last_usage(self) -> None:
        """
        Populate node.last_usage recursively (also for the nodes within a FusedSchedulerNode)
        """
    def free_buffers(self) -> None:
        """Free any buffers that are no longer needed"""
    def flush(self) -> None: ...
    def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode) -> None: ...
    def create_backend(self, device: torch.device) -> BaseScheduling: ...
    def get_backend(self, device: torch.device | None) -> BaseScheduling: ...
    def enter_context(self, node: BaseSchedulerNode) -> None: ...
    def can_buffer_be_removed_through_fusion(self, name: str, fused_node_names: OrderedSet[str]) -> bool: ...
    def should_partition(self, node: BaseSchedulerNode) -> bool:
        """Return True if we should partition the inductor graph on this node"""
    def get_name_to_nodes(self) -> dict[str, ir.IRNode | ir.TorchBindObject | sympy.Expr]:
        """
        Return a mapping from name strings to the corresponding graph inputs or
        base scheduler node outputs.
        """
    def compute_graph_partition_maps(self, signatures: list[GraphPartitionSignature]) -> None:
        """
        computes a mapping from partition input/output indices to graph input/output
        indices for each partition.
        """
    def get_graph_partition_symbol_inputs(self, partition: PartitionType, input_nodes: dict[str, ir.IRNode | ir.TorchBindObject | sympy.Expr]) -> OrderedSet[sympy.Symbol]:
        """
        Returns all symbol inputs which are required to be in scope to successfully
        perform codegen for this graph partition, including:
        - free symbols used in partition nodes
        - free symbols in partition input/node shapes, strides, and offsets. This is needed
          for recording cudagraphs for tensors with dynamic shapes.
        """
    def get_graph_partition_signature(self, partitions: list[PartitionType], skip_cudagraphs: list[bool]) -> list[GraphPartitionSignature]:
        """
        Gets signature for each graph partition, including input nodes, output nodes, and
        whether deallocating an input within graph partition.
        """
    def clean_removed_buffer_from_partition_signatures(self, signature: GraphPartitionSignature) -> GraphPartitionSignature:
        """
        Updates the partition signature by removing buffers specified in
        V.graph.removed_buffers. See [Note: Removed Graph Partition Arguments]
        """
    def reorder_for_minimizing_partition(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Reorder nodes to minimize the number of partitions via a bfs
        topological sort. This is the optimal reordering such that the
        number of partitions cannot be reduced further. This may be
        sub-optimal for other metrics such as peak memory. This does not
        change relative orders of two cudagraphable nodes, nor the
        relative order of two non_cudagraphable nodes.
        """
    def maybe_reorder_for_minimizing_partition(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Reorder nodes to minimize the number of partitions if this only slightly
        increase peak memory.
        """
    def reorder_for_partition_with_simple_dependency(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """
        Reorder a node if it should be partitioned and has simple dependency:
        1. move a partitioned node to the front if it has no dependency
        2. move a partitioned node to the back if it is only used by OutputNode
        3. otherwise do not reorder
        """
    def graph_partition(self) -> tuple[list[PartitionType], list[GraphPartitionSignature]]:
        """
        Given a list of BaseSchedulerNodes, split into a list of
        graph partitions and compute partition input/output signatures.
        """
    def codegen(self) -> None: ...
    def _codegen_partition_wrapper(self, partition: PartitionType, signature: GraphPartitionSignature) -> None:
        """Codegen a partition given its inputs/outputs"""
    def _codegen_partitions(self) -> None:
        """
        Split nodes into partitions and codegen each partition into separate functions.
        This allows further applying different optimizations (e.g., cudagraph) to
        each function.
        """
    def _codegen(self, nodes: list[BaseSchedulerNode]) -> None: ...
    def benchmark_combo_kernel(self, node_list: Sequence[BaseSchedulerNode]) -> tuple[float, float, list[str | None]]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
    def speedup_by_combo_kernel(self, nodes: list[BaseSchedulerNode]) -> bool:
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """
    def get_buffer_layout(self, buf_name: str) -> ir.Layout: ...
    def update_zero_dim_cpu_tensor(self) -> None: ...

class BaseScheduling:
    scheduler: Incomplete
    def __init__(self, scheduler: Scheduler | None) -> None: ...
    def free_buffers_in_scheduler(self) -> None: ...
    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        """Return a set of .codegen.common.BackendFeature()"""
    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
    def can_fuse_multi_outputs_template(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        A Multi-Output Template (referenced in #144012) is a template node
        with MultiOutputLayout, and its output buffers are instances of MultiOutput.
        In this context, we verify whether node1 represents the Multi-Output Template
        and node2 corresponds to one of its outputs. If so, we further check if
        backend supports this fusion.
        """
    def fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> FusedSchedulerNode:
        """
        Fuse two nodes
        """
    def group_fn(self, sizes: Sequence[Sequence[sympy.Expr]]) -> tuple[tuple[sympy.Expr, ...], ...]:
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
    def codegen_template(self, template_node: BaseSchedulerNode, epilogue_nodes: Sequence[BaseSchedulerNode], prologue_nodes: Sequence[BaseSchedulerNode]) -> str | None:
        """
        Given a template node, generate a kernel.

        This function is only available for triton now. If the third-party backend behaves as a sub-class
        of TritonScheduling, it can override it or reuse it.
        """
    def generate_kernel_code_from_nodes(self, nodes: Sequence[BaseSchedulerNode], benchmark_kernel: bool) -> str:
        """
        Generate a kernel given a list of pre-fused nodes.
        """
    def codegen_node(self, node: FusedSchedulerNode | SchedulerNode) -> None:
        """
        Generate a kernel given a list of pre-fused nodes.
        """
    def codegen_sync(self) -> None:
        """
        Generate synchronization code for the kernel. This method depends on the hardware characteristics.
        """
    def ready_to_flush(self) -> bool:
        """
        Check whether the backend is requesting the scheduler to flush the generated kernel.
        If not supported, please return False.
        """
    def flush(self) -> None:
        """
        Flush the generated kernel and python wrapper code to the source code file.
        """
    def benchmark_fused_nodes(self, nodes: Sequence[BaseSchedulerNode]) -> tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
    def benchmark_codegened_module(self, module: ModuleType) -> tuple[float, str]:
        """
        Benchmark a compiled module and return the execution time
        in milliseconds on randomly generated inputs.
        """
    def get_fusion_pair_priority(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> int:
        """
        Return an unsigned integer which represents the priority of this fusion pair.
        The smaller is with higher priority.
        """
    def benchmark_combo_kernel(self, node_list: Sequence[BaseSchedulerNode]) -> tuple[float, float, list[str | None]]:
        """
        Benchmark the list of nodes to combine and return the execution time
        and memory copy time in milliseconds on randomly generated inputs.
        """
