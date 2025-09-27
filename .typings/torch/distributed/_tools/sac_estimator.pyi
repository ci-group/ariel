import torch
from _typeshed import Incomplete
from collections import OrderedDict
from dataclasses import dataclass
from torch import UntypedStorage, nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import SAC_IGNORED_OPS
from typing import Any, NamedTuple
from typing_extensions import Self

__all__ = ['SACEstimator', 'SACStats', 'MSPS', 'SACTradeOffStats', 'SACGreedyOrderMeta']

OPS_TO_ALWAYS_SKIP = SAC_IGNORED_OPS | _ADDITIONAL_IGNORED_OPS

@dataclass
class _SACMetadata:
    """
    Stores metadata for a single operator for SAC.

    Attributes:
        func (Any): The operator function.
        time_taken (float): The time taken by the operator.
        memory_used (float): The memory used by the operator.
        curr_idx (int): The current operator index.
        output_ids (Tuple[int, ...]): The storage IDs of the operator's outputs.
        inplace_info (Tuple[int, ...]): Tuple of self and parent operator for in-place operator.
        is_view_like (bool): Whether the operator is view-like.
        is_rand_op (bool): Whether the operator is a random operator.
    """
    func: Any
    time_taken: float
    memory_used: float
    curr_idx: int
    output_ids: tuple[int, ...]
    inplace_info: tuple[int, ...]
    is_view_like: bool
    is_rand_op: bool

@dataclass
class _SACModMetadata:
    """
    Stores metadata for a module for SAC.

    Attributes:
        start_idx (int): The starting index of the module's operators.
        force_store_random (bool): Whether to force store random operators in the module.
        sac_metadata (List[_SACMetadata]): List of metadata for each operator in the module.
    """
    start_idx: int
    force_store_random: bool
    sac_metadata: list[_SACMetadata]

@dataclass
class SACStats:
    """
    A class for storing Activation Checkpointing statistics corresponding to a module.

    Attributes:
        func_names (List[str]): List of operator names.
        runtimes (List[float]): List of operator runtimes in millliseconds.
        memory (List[int]): List of operator memory usage in bytes.
        view_like_ops (List[int]): Indices of view-like operators.
        rand_ops (List[int]): Indices of random operators.
        saved_autograd_ops (List[int]): Indices of operator results saved by autograd engine.
        inplace_ops (List[Tuple[int, int]]): Tuple of indices of op and its first parent for Inplace operators.
        force_store_random (bool): Whether to force store random operator results.
    """
    func_names: list[str]
    runtimes: list[float]
    memory: list[int]
    view_like_ops: list[int]
    rand_ops: list[int]
    saved_autograd_ops: list[int]
    inplace_ops: list[tuple[int, int]]
    force_store_random: bool

class MSPS(NamedTuple):
    """
    Represents Memory and Runtime Statistics for an operator/operator group.

    Attributes:
        func_names (set[str]): Set of operator/operator group names.
        op_idx (int): Operator index (group head index in case of operator groups).
        memory (int): Memory usage in bytes.
        runtime (float): Runtime in milliseconds.
        msps (float): Memory per second calculated as memory/runtime.
    """
    func_names: set[str]
    op_idx: int
    memory: int
    runtime: float
    msps: float

@dataclass
class SACTradeOffStats:
    """
    Stores statistics for activation-checkpointing trade-off.

    Attributes:
        n_segments (int): Number of piecewise linear segments fitted to the trade-off curve.
        slopes (List[float]): Slopes of the pieces of linear segments fitted to the trade-off curve.
        intercepts (List[float]): Intercepts of the of the pieces of linear segments fitted to the trade-off curve.
        fit_breaks (List[float]): Breakpoints of the of the pieces of linear segments fitted to the trade-off curve.
        tradeoff_curve (OrderedDict[float, float]): Trade-off curve data of memory discarded vs recomputation time.
        sac_memory (int): Total memory of operations available for activation checkpointing in bytes.
        sac_runtime (float): Total runtime of operations available for activation checkpointing in milliseconds.
    """
    n_segments: int
    slopes: list[float]
    intercepts: list[float]
    fit_breaks: list[float]
    tradeoff_curve: OrderedDict[float, float]
    sac_memory: int
    sac_runtime: float

@dataclass
class SACGreedyOrderMeta:
    """
    Stores metadata for Greedy-order SAC.

    Attributes:
        recomputed_ops (set[int]): Set of operator indices to be recomputed.
        stored_ops (set[int]): Set of operator indices to be stored.
        inplace_op_groups (dict[int, set[int]]): Dictionary of inplace operator groups from group-head to operators.
        random_ops_group (dict[int, set[int]]): Dictionary of random op group head to random ops.
        msps_meta (list[MSPS]): List of Memory and Runtime Statistics for operators.
    """
    recomputed_ops: set[int]
    stored_ops: set[int]
    inplace_op_groups: dict[int, set[int]]
    random_ops_group: dict[int, set[int]]
    msps_meta: list[MSPS]

class SACEstimator(TorchDispatchMode):
    '''
    Estimates the memory and recomputation time trade-offs for applying Selective Activation Checkpointing (SAC).

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the memory and
    runtime trade-offs of functions or ``torch.nn.Module``s for Selective Activation Checkpointing (SAC). It provides
    detailed statistics and metadata information for operators of each module and provides a greedy order for selecting
    the operators to be recomputed/checkpointed.  It also constructs the per-module trade-off graph of discarded memory
    vs recomputation time for the obtained greedy order. Using ``RuntimeEstimator`` under the hood, it supports two
    estimation modes, `operator-level-benchmark` and (`operator-level-cost-model` (roofline model).

    Attributes:
        sac_mod_stats (Dict[str, SACStats]): Dictionary from module FQN (fully qualified name) to ``SACStats``.
        sac_mod_tradeoff_stats (Dict[str, SACTradeOffStats]): Dictionary from module FQN to ``SACTradeOffStats``.
        sac_mod_greedy_order_meta (Dict[str, SACGreedyOrderMeta]): Dictionary from module FQN to ``SACGreedyOrderMeta``.

    Note:
        1) This class is designed to be used under ``FakeTensorMode``.
        2) Currently, it only supports estimation of compute time and memory usage, and does not consider communication.

    Example usage:

        .. code-block:: python

            sac_estimator = SACEstimator()
            with FakeTensorMode():
                module = ...
                inp = ...
                with sac_estimator("operator-level-cost-model"):
                    output = module(inp)
                sac_estimator.display_modulewise_sac_stats(depth=4, print_tabular=True)
    '''
    sac_mod_stats: dict[str, SACStats]
    sac_mod_tradeoff_stats: dict[str, SACTradeOffStats]
    sac_mod_greedy_order_meta: dict[str, SACGreedyOrderMeta]
    _mod_tracker: Incomplete
    _sac_metadata: list[_SACMetadata]
    _sac_mod_metadata: dict[str, _SACModMetadata]
    _leaf_modules: set[str]
    _saved_tensor_hook_ctx: Incomplete
    _saved_tensor_ids: set[int]
    _estimate_runtime: Incomplete
    def __init__(self) -> None: ...
    def _pack_hook(self, x: torch.Tensor) -> torch.Tensor: ...
    def _pre_fw_hook(self, mod: nn.Module, inputs: Any) -> None: ...
    def _post_fw_hook(self, mod: nn.Module, inputs: Any, outputs: Any) -> None: ...
    def _get_force_store_random(self, inputs: Any) -> bool: ...
    def _get_sac_stats(self, data: list[_SACMetadata], force_store_random: bool) -> SACStats: ...
    def _get_inplace_metadata(self, func: Any, out_storages: set[UntypedStorage]) -> tuple[int, tuple[int, ...], dict[str, tuple[int, ...]]]: ...
    def __torch_dispatch__(self, func, types, args=..., kwargs=None): ...
    def _get_greedy_order_meta(self, sac_stats: SACStats) -> SACGreedyOrderMeta: ...
    def _get_sac_tradeoff_pwlf_stats(self, sac_stats: SACStats, greedy_order_meta: SACGreedyOrderMeta, n_segments: int = 2, save_tradeoff_graph: bool = False, filename: str = 'ac_tradeoff') -> SACTradeOffStats: ...
    def display_sac_stats(self, sac_stats: SACStats, print_tabular: bool = False) -> None:
        """
        Displays the SAC statistics.

        Args:
            sac_stats (SACStats): The SAC statistics to display.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            1. Total Memory: The total memory usage in bytes.
            2. Total Runtime: The total runtime in milliseconds.
            3. Store Random: A flag indicating whether to force store random operator results.

            Followed by a table with the following columns:
            1. Op Idx: The operator index.
            2. Op Name: The operator name.
            3. Runtimes (ms): The operator runtime in milliseconds.
            4. Memory (B): The operator memory usage in bytes.
            5. View-like: A flag indicating whether the operator is view-like.
            6. Random: A flag indicating whether the operator is random.
            7. Saved Autograd: A flag indicating whether the operator's result is saved by autograd engine.
            8. In-place: The index of the operator's first parent, or None if not in-place.

        If print_tabular is True, the table is printed in a tabular format.
        Otherwise, the table is printed in a plain text format.
        """
    def display_sac_tradeoff_stats(self, greedy_order_meta: SACGreedyOrderMeta, sac_stats: SACStats, print_tabular: bool = False) -> None:
        """
        Displays the SAC trade-off statistics.

        Args:
            greedy_order_meta (SACGreedyOrderMeta): The SAC greedy order metadata.
            sac_stats (SACStats): The SAC statistics.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            A table with the following columns:
            1. Op Id(s): The operator index(es).
            2. Op Name(s): The operator name(s).
            3. Discarded Mem (%): The percentage of discarded memory.
            4. Discarded Mem (B): The discarded memory in bytes.
            5. Recomp time (%): The percentage of recomputed time.
            6. Recomp time (ms): The recomputed time in milliseconds.
            7. MSPS: The memory per second.
            8. Always Stored: A flag indicating whether the operator is always stored.
            9. Always Recomputed: A flag indicating whether the operator is always recomputed.

        If print_tabular is True, the table is printed in a tabular format.
        Otherwise, the table is printed in a plain text format.
        """
    def pwlf_sac_tradeoff_curve(self, n_segments: int = 2, save_tradeoff_graphs: bool = False) -> None:
        """
        Fits a piecewise linear function with the specified sumber of segments to the SAC trade-off curve of
        discarded memory vs recomputation time.

        Args:
            n_segments (int, optional): The number of segments to be used for fitting the piecewise linear function to
                the trade-off curve. Defaults to 2.
            save_tradeoff_graphs (bool, optional): Whether to save the trade-off graphs to file. Defaults to False.

        If save_tradeoff_graphs is True, the trade-off graphs are saved to file using the module FQN as the filename.
        """
    def display_modulewise_sac_stats(self, depth: int = 2, print_tabular: bool = False) -> None:
        """
        Displays the SAC and trade-off statistics for each module.

        Args:
            depth (int, optional): The maximum depth of modules to display. Defaults to 2.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            For each module with depth less than or equal to the specified depth:
            1. The SAC statistics for the module (using display_sac_stats).
            2. The SAC trade-off statistics for the module (using display_sac_tradeoff_stats).

        If print_tabular is True, the statistics are printed in a tabular format.
        Otherwise, the statistics are printed in a plain text format.
        """
    def __call__(self, estimate_mode_type: str) -> Self:
        '''
        Sets the estimate mode type.

        Currently supported modes:
            - "operator-level-benchmark": Estimates runtime using operator benchmarking.
            - "operator-level-cost-model": Estimates runtime using roofline cost model.

        Args:
            estimate_mode_type (str): The type of estimate mode to use.

        Returns:
            SACEstimator: The SAC estimator instance.

        Raises:
            NotImplementedError: If the estimate mode type is not supported.
        '''
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: Any) -> None: ...
