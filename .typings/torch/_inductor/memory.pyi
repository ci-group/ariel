import dataclasses
from .dependencies import Dep as Dep
from .ir import MultiOutputLayout as MultiOutputLayout, NoneLayout as NoneLayout
from .scheduler import BaseSchedulerNode as BaseSchedulerNode, SchedulerBuffer as SchedulerBuffer
from .utils import get_dtype_size as get_dtype_size, is_wait as is_wait
from .virtualized import V as V
from _typeshed import Incomplete
from torch._utils_internal import signpost_event as signpost_event
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Callable

torch_log: Incomplete

@dataclasses.dataclass
class PeakMemoryResult:
    order: list[BaseSchedulerNode]
    peak_memory: int
    method: str

@dataclasses.dataclass
class MemoryPlanningInfoForBuffer:
    size_alloc: int = ...
    size_free: int = ...
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(default_factory=OrderedSet)

@dataclasses.dataclass
class MemoryPlanningInfoForNode:
    index: int = ...
    size: int = ...
    pred_buffers: OrderedSet[SchedulerBuffer | FreeableInputBuffer] = dataclasses.field(default_factory=OrderedSet)
    pred_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(default_factory=OrderedSet)
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(default_factory=OrderedSet)

@dataclasses.dataclass
class FreeableInputBuffer:
    name: str
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(default_factory=MemoryPlanningInfoForBuffer)
    def get_name(self) -> str: ...
    def __hash__(self) -> int: ...

def get_freeable_input_buf(nodes: list[BaseSchedulerNode], graph_inputs: OrderedSet[str]) -> dict[str, FreeableInputBuffer]:
    """
    Create and keep track of all input buffers that can be freed during the program

    Returns:
        A dictionary containing all freeble input buffers, keyed by their names.
    """
def compute_size_for_scheduler_buffer(name_to_buf: dict[str, SchedulerBuffer]) -> dict[str, tuple[int, int]]:
    """
    Compute the size of each scheduler buffer, including (1) memory allocated when
    it is created and (2) memory deallocated when it is freed.

    We specially handle the case of MultiOutputLayout.
    Consider the following case:
        buf0 = some_ops_with_multi_outputs(...)
        buf1 = buf0[0] # assume 10 bytes
        buf2 = buf0[1] # assume 20 bytes
    In such cases,
        buf0: at creation, 30 bytes allocated, when deleted, 0 bytes freed
        buf1: at creation, 0 bytes allocated, when deleted, 10 bytes freed
        buf2: at creation, 0 bytes allocated, when deleted, 20 bytes freed

    Returns:
        A dictionary mapping a scheduler buffer to a tuple of (size_alloc, size_free).
    """
def assign_memory_planning_info_for_scheduler_buffers(nodes: list[BaseSchedulerNode], name_to_buf: dict[str, SchedulerBuffer]) -> None:
    """
    For each SchedulerBuffer, assign its size info and successor nodes.
    A buffer's successor nodes determines when a buffer can be freed.
    """
def assign_memory_planning_info_for_scheduler_nodes(nodes: list[BaseSchedulerNode], name_to_fused_node: dict[str, BaseSchedulerNode], name_to_buf: dict[str, SchedulerBuffer], name_to_freeable_input_buf: dict[str, FreeableInputBuffer]) -> None:
    """
    Assign to each scheduler node its predecessor and successor nodes.
    """
def estimate_peak_memory(nodes: list[BaseSchedulerNode], name_to_freeable_input_buf: dict[str, FreeableInputBuffer], graph_outputs: OrderedSet[str]) -> tuple[int, list[int]]:
    """
    Given a list of nodes in their execution order, estimate the peak memory, by
    keeping track of the liveliness of SchedulerBuffers and FreeableInputBuffers.

    Returns:
        int: peak memory
        List[int]: memory usage at each node (or each step).
    """
def topological_sort_lpmf(nodes: list[BaseSchedulerNode], name_to_freeable_input_buf: dict[str, FreeableInputBuffer], name_to_buf: dict[str, SchedulerBuffer], graph_outputs: OrderedSet[str]) -> list[BaseSchedulerNode]:
    '''
    A bfs-based greedy topological order. LPMF stands for "Least Peak Memory First".

    The idea is from this paper:
    Buffer memory optimization for video codec application modeled in Simulink
    https://www.cs.york.ac.uk/rts/docs/DAC-1964-2006/PAPERS/2006/DAC06/PDFFILES/P0689.PDF

    The algorithm maintain the max memory so far.
    At every iteration, for each scheduleable node, it computes:
        - how much memory needs to be allocated for the output buffers of this node;
        - how much memory can be freed as a result of executing this node.
    This gives us two values for each node:
        (1) mem1: memory during the execution of the node;
        (2) mem2: memory after executing the node, after some input buffers are freed.
    The greedy approach select as follows:
        (i) if there are nodes whose mem1 values are below the max memory so far,
            then pick the node with the lowest mem2 value;
        (ii) otherwise, pick the one with the lowest mem1 value.
    '''
def topological_sort_bfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    A BFS topological sort that selects nodes whose dependencies are executed the
    earliest. This follows a FIFO idea. Specifically, at every iteration, for each node
    that is schedulable, we gather the order in which its predecessor nodes are executed,
    and this sorted list of execution orders of predecessor nodes defines the priority.
    We select the node whose predecessors nodes are executed the earliest. The FIFO
    idea aims to reduce the liveness duration of buffers created.
    """
def topological_sort_dfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    This is a DFS topological sort. The setup is similar to `topological_sort_schedule`
    in scheduler.py. The difference is the order nodes are visited in the outer loop.
    In `topological_sort_schedule`, nodes are visited in their original order.
    In this function, nodes are visited based on their priority -- for each node, we
    compute the total memory of all buffers it reads from or writes to, and we visit
    the nodes in ascending order of this priority.
    """
def prepare_planning_info(nodes: list[BaseSchedulerNode], name_to_buf: dict[str, SchedulerBuffer], name_to_fused_node: dict[str, BaseSchedulerNode], graph_inputs: OrderedSet[str], graph_outputs: OrderedSet[str]) -> tuple[int, dict[str, FreeableInputBuffer]]:
    """
    Prepare planning info. As nodes are scheduled one at a time, these help
    keep track of when a buffer can be freed, and when a node can be scheduled

    Returns:
        int: peak memory estimation
        dict[str, FreeableInputBuffer]: name to freeable input buffer
    """
def reorder_for_peak_memory(nodes: list[BaseSchedulerNode], name_to_buf: dict[str, SchedulerBuffer], name_to_fused_node: dict[str, BaseSchedulerNode], graph_inputs: OrderedSet[str], graph_outputs: OrderedSet[str], methods: list[Callable[..., list[BaseSchedulerNode]]] = ...) -> list[BaseSchedulerNode]:
    """
    Try a few heuristics based topological sort algorithms, and pick the one whose
    resulting topological order has the lowest peak memory estimation.
    """
