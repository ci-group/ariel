import torch
from _typeshed import Incomplete
from collections import OrderedDict
from torch.distributed._tools.mem_tracker import MemTracker as MemTracker, _MemRefType as _MemRefType, _ModMemStats as _ModMemStats, _ModState as _ModState
from torch.distributed._tools.runtime_estimator import RuntimeEstimator as RuntimeEstimator
from torch.distributed._tools.sac_estimator import SACEstimator as SACEstimator, SACTradeOffStats as SACTradeOffStats
from typing import TypedDict

class ModOrder(TypedDict):
    fw_pre_order: list[str]
    bw_pre_order: list[str]
    fw_post_order: list[str]
    bw_post_order: list[str]

class ModRuntime(TypedDict):
    fw: float
    bw: float

class ModStats(TypedDict):
    fqn: str
    param_per_module: int
    grad_per_module: int
    grad_total: int
    act_fw_per_module: int
    act_bw_per_module: int
    act_grad_per_module: int
    act_total: int
    input_per_module: int
    output_per_module: int
    fw_runtime_per_module: float
    bw_runtime_per_module: float
    is_leaf: bool
    sac_runtime: float
    sac_memory: int
    n_segments: int
    slopes: list[float]
    intercepts: list[float]
    breakpoints: list[float]
    tradeoff_curve: OrderedDict[float, float]

class ModuleInfo(TypedDict):
    mod_order: ModOrder
    mod_stats: list[ModStats]

def aggregate_stats(model: torch.nn.Module, mem_tracker: MemTracker, runtime_estimator: RuntimeEstimator, sac_estimator: SACEstimator, dev: torch.device) -> ModuleInfo:
    """
    Collect modulewise stats for a given model, including memory, runtime, and AC tradeoff stats.

    Args:
        model: nn.Module object
        runtime_estimator: RuntimeEstimator object with runtime stats
        mem_tracker: MemTracker object with memory stats
        sac_estimator: SACEstimator object with AC tradeoff stats
        dev: device the model was run on (used to extract memory stats from MemTracker)

    Returns:
        ModuleInfo: A dictionary with module order and module stats.
    """

class Node(ModStats):
    index: int
    pos_fw_post_order: int

class Graph:
    nodes: list[Node]
    name2node: dict[str, Node]
    ad_matrix: Incomplete
    fw_post_order: list[str]
    def __init__(self, n: int) -> None: ...
    def add_node(self, node: Node) -> None: ...

def parse_module_info(module_info: ModuleInfo) -> Graph:
    """
    Parse module info and create a graph (tree) of modules. The graph will be
    used by MILP solver to find optimal SAC and/or FSDP configurations.
    """
def is_self_or_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    check if name_descendant is a submodule of name_ancestor, or if they are the same
    """
def is_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    if name_descendant is a submodule of name_ancestor, but not the same
    """
def display_bytes(b: int, unit: str = 'MiB') -> str:
    """
    return a string that represent the number of bytes in a desired unit
    """
def get_peak_memory_runtime_baseline(graph: Graph) -> tuple[int, float]:
    """
    Get the baseline peak memory and runtime.
    Baseline here means there is no FSDP or AC.
    Memory includes the parameters, gradients, activations, and activation gradients.
    Memory does not include e.g., optimizer states, embedding tables, etc.

    Returns:
        int: peak memory in bytes
        float: compute time in ms
    """
