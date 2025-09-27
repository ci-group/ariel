from _typeshed import Incomplete
from enum import IntEnum
from torch.distributed._tools.ilp_utils import Graph as Graph, is_submodule as is_submodule
from torch.distributed._tools.sac_estimator import SACStats as SACStats

logger: Incomplete

def sac_milp(graph: Graph, memory_budget: float, world_size: int = 1, ac_units: list[str] | None = None, fsdp_units: list[str] | None = None) -> tuple[dict[str, float], float, int]:
    """
    MILP to decide which modules to AC and how much memory to discard.
    The objective is to minimize recomputation time.
    The constraint is to ensure peak memory is under budget.

    Args:
        graph: graph representation of the model as a module submodule tree
            where each node is a submodule with memory & runtime stats
        memory_budget: memory budget in GiB
        world_size: number of GPUs. In the case of FSDP, world_size will be
            used to compute the amount of parameter and gradient memory on each rank
        ac_units: a list of user-specified AC units.
        fsdp_units: a list of FSDP units. AC units cannot be supermodules of FSDP units.

    Returns:
        Dict[str, float]: the optimal SAC solution, mapping from module fqn to
            the percentage of activation memory to **discard**
        float: the recomputation time of the optimal SAC solution
        int: upper bound on the peak memory of the optimal SAC solution.
            note that value of -1 means that the ILP solver failed to find a solution.

    """

class SACDecision(IntEnum):
    RECOMPUTE = 0
    SAVE = 1

def get_optimal_checkpointing_policy_per_module(sac_stats: SACStats, memory_budget: float) -> list[int]:
    """
    This is adapted from --
    https://github.com/facebookresearch/xformers/blob/c6c0ac31f1b08542a0bc27278c6ed10f825f6963/xformers/checkpoint.py#L375

    Given the SACStats of a module, including list of operators, their memory, runtimes, and metadata,
    decide via MILP an optimal set of operators to checkpoint under a given ``memory_budget``.

    Args:
        sac_stats: the SACStats object of the module
        memory_budget: a float between zero and one

    Returns:
        List[int]: the decision whether each operator should be saved (1) or recomptued (0).
    """
