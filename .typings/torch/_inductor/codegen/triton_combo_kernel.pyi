import sympy
from .. import config as config, metrics as metrics
from ..runtime.hints import DeviceProperties as DeviceProperties
from ..runtime.runtime_utils import next_power_of_2 as next_power_of_2
from ..runtime.triton_heuristics import RoundRobinComboKernelGrid as RoundRobinComboKernelGrid, SequentialComboKernelGrid as SequentialComboKernelGrid
from ..scheduler import BaseSchedulerNode as BaseSchedulerNode
from ..utils import Placeholder as Placeholder, triton_version_uses_attrs_dict as triton_version_uses_attrs_dict
from ..virtualized import V as V
from .common import ArgName as ArgName, ConstexprArg as ConstexprArg, DeferredLine as DeferredLine, IndentedBuffer as IndentedBuffer, InplacedBuffer as InplacedBuffer, Kernel as Kernel, PythonPrinter as PythonPrinter, RemovedArg as RemovedArg, SizeArg as SizeArg, WorkspaceArg as WorkspaceArg
from .simd import SIMDScheduling as SIMDScheduling, prefix_is_reduction as prefix_is_reduction
from .simd_kernel_features import SIMDKernelFeatures as SIMDKernelFeatures
from .triton import TritonKernel as TritonKernel, gen_common_triton_imports as gen_common_triton_imports
from .triton_utils import config_of as config_of, signature_to_meta as signature_to_meta
from _typeshed import Incomplete
from dataclasses import dataclass
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable

log: Incomplete
pexpr: Incomplete
LARGE_NUMELS: float
BLOCK_UTILIZATION: float

def _default_custom_combo_kernel_horizontal_partition(nodes: list[BaseSchedulerNode], triton_scheduling: SIMDScheduling, kernel_map: dict[BaseSchedulerNode, TritonKernel], node_info_map: dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]]) -> list[list[BaseSchedulerNode]]:
    """Horizontally partition the given list of nodes into a list of list of nodes where each sublist
    represents a partition. Nodes in different partitions are implemented in different combo kernels.
    Nodes in the same partition are likely to be implemented
    in the same combo kernel, but subject to subsequent restrictions like CUDA limits for number of args.

    Input arguments:
        nodes: a list of fused scheduler nodes to partition.
        triton_scheduling: TritonScheduling instance.
        kernel_map: a map from node to its kernel.
        node_info_map: a map from node to (node_schedule, tiled_groups, numel, rnumel).
    Output:
        a list of list of nodes with each sublist representing a partition.

    The default algorithm is to partition nodes based on the following rules:
        1) nodes with the same number of block dimensions are grouped together.
        2) large pointwise nodes (numels greater than LARGE_NUMELS) are separated from other nodes.
        3) large reduce nodes are separated from other nodes.
    """

_custom_combo_kernel_horizontal_partition_algorithm: Callable[[list[BaseSchedulerNode], SIMDScheduling, dict[BaseSchedulerNode, TritonKernel], dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]]], list[list[BaseSchedulerNode]]]

def set_custom_combo_kernel_horizontal_partition(algorithm: Callable[[list[BaseSchedulerNode], SIMDScheduling, dict[BaseSchedulerNode, TritonKernel], dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]]], list[list[BaseSchedulerNode]]]) -> None:
    """Sets the algorithm used to partition nodes into horizontal partitions. Nodes in different partitions
    are implemented in different combo kernels. Nodes in the same partition are likely to be implemented
    in the same combo kernel, but subject to subsequent restricts like CUDA limits for number of args.

    The algorithm should take a list of nodes and return a list of list of nodes.

    The default algorithm is to partition nodes based on number of block dimensions.
    """

@dataclass
class PartitionState:
    partitions: list[list[BaseSchedulerNode]]
    cur_partition: list[BaseSchedulerNode]
    cur_count: int
    def finalize(self) -> None: ...

class ComboKernel(Kernel):
    MAX_NUM_ARGS: int
    @staticmethod
    def _update_partition(partition_state: PartitionState, node_rw_count: int, node_info: BaseSchedulerNode) -> None: ...
    @staticmethod
    def _base_horizontal_partition(subkernel_nodes: list[BaseSchedulerNode], triton_scheduling: SIMDScheduling, node_info_map: dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]], custom_algorithm: bool) -> list[list[BaseSchedulerNode]]:
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
    @staticmethod
    def horizontal_partition(nodes: list[BaseSchedulerNode], triton_scheduling: SIMDScheduling, kernel_map: dict[BaseSchedulerNode, TritonKernel], node_info_map: dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]], custom_algorithm: bool = False) -> list[list[BaseSchedulerNode]]:
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnum)
        for each subkernel node where each sublist forms a ComboKernel. It horizontally partitions nodes into
        sublists in the following way:
            1) call _custom_combo_kernel_horizontal_partition_algorithm() if custom_algorithm is True
            2) then, call _base_horizontal_partition() to partition nodes into sublists, each sublist is
               guaranteed to not exceed CUDA limits for number of args (read/writes) and to have the same
               2D or 1D blocking strategy.
        """
    class SequentialDispatch:
        """
        The dispatcher which dispatches the subkernels in a sequential manner:
        the blocks are first dispatched to the 1st subkernel (until it is filled),
        then to the 2nd subkernel, and so on.
        The class defines the methods specific to the dispatch algorithm.
        Methods:
            codegen_pid_range(...): codegen the pid range for each subkernel.
            grid(...): codegen the grid size for launching the combo kernel.
        """
        grid_expr = SequentialComboKernelGrid
        @classmethod
        def codegen_pid_range(cls, kernel: ComboKernel, num: int, code: IndentedBuffer) -> None: ...
        @classmethod
        def _calculate_xblocks(cls, kernel: ComboKernel, code: IndentedBuffer) -> None: ...
    class RoundRobinDispatch:
        """
        The dispatcher which dispatches the subkernels in a round robin manner:
        the blocks are interleavedly dispatched to each subkernel to execute them
        in parallel.
        The class defines the methods specific to the dispatch algorithm.
        Methods:
            codegen_pid_range(...): codegen the pid range for each subkernel.
            grid(...): codegen the grid size for launching the combo kernel.
        """
        grid_expr = RoundRobinComboKernelGrid
        @classmethod
        def codegen_pid_range(cls, kernel: ComboKernel, num: int, code: IndentedBuffer) -> None: ...
    sub_kernels: list[TritonKernel]
    iter_vars_count: Incomplete
    grids: list[list[int]]
    min_x_blocks_list: list[int | str]
    x_numels_list: list[int | str]
    enable_autotune: Incomplete
    mixed_sizes: Incomplete
    dispatch_class: type[ComboKernel.SequentialDispatch | ComboKernel.RoundRobinDispatch] | None
    block_args: list[str]
    block_size_1d: int
    block_size_2d: int
    num_warps: int
    block_size_reduce: int
    dynamic_shape_args: list[str]
    def __init__(self, enable_autotune: bool = False, mixed_sizes: bool = False) -> None: ...
    def create_sub_kernel(self, triton_kernel: TritonKernel) -> TritonKernel: ...
    @staticmethod
    def create_triton_kernel(tiling: dict[str, sympy.Expr], features: SIMDKernelFeatures, optimize_mask: bool) -> TritonKernel:
        """
        Only allow optimize_mask=True when 1) sequential dispatch is used,
        2) numels except x dimension are the same for each sub kernel.
        """
    def codegen_static_numels_sub_kernel(self, code: IndentedBuffer, sub_kernel: TritonKernel, num: int) -> list[str]:
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        rnumel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
    def min_x_blocks_sub_kernel(self, sub_kernel: TritonKernel, num: int) -> None:
        """
        Kernels with no_x_dim being true has no tunable XBLOCK. They have a fixed number of X blocks.
        Grid calculation needs to make sure that they are assigned with enough number of blocks.
        """
    def select_heuristics(self, sub_kernel: TritonKernel) -> tuple[str, dict[str, int]]: ...
    def select_combo_heuristics(self, heuristics_list: list[str], size_hints_list: list[dict[str, int]]) -> tuple[str, dict[str, int], TritonKernel]: ...
    def get_mutated_args_sub_kernels(self) -> list[str]: ...
    def select_dispatch_strategy(self) -> None: ...
    def jit_line(self, heuristics: str, size_hints: dict[str, int], selected_kernel: TritonKernel, signature: list[Any], argdefs: list[ArgName], pointwise_with_reduce: bool = False) -> str: ...
    def codegen_blocks(self, code: IndentedBuffer) -> None: ...
    def get_block_args(self) -> list[ConstexprArg]:
        """
        Calculate blocks from sub_kernels and range_trees.
        **Update self.block_args**
        Return the block args
        """
    def add_numel_to_args(self, argdefs: list[ArgName], signature: list[Any]) -> list[ArgName]: ...
    def add_numel_to_call_args(self, name: str, call_args: list[Any], arg_types: list[Any]) -> None: ...
    def kernel_benchmark_extra_args(self) -> list[str]: ...
    def codegen_kernel(self, name: str | None = None) -> str: ...
    def codegen_kernel_benchmark(self, num_gb: float) -> IndentedBuffer: ...
    def imports_for_benchmark_kernel(self) -> str: ...
    def uniquify_block_sizes(self, code: IndentedBuffer, num_kernel: int, uniquify: list[str]) -> IndentedBuffer: ...
    def call_kernel(self, code: IndentedBuffer, name: str) -> None: ...
    def combo_grid_meta(self) -> dict[str, Any]: ...
