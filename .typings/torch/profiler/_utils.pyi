from _typeshed import Incomplete
from dataclasses import dataclass
from torch.autograd import _KinetoEvent as _KinetoEvent
from torch.autograd.profiler import profile as profile
from torch.profiler import DeviceType as DeviceType

def _traverse(tree, next_fn, children_fn=..., reverse: bool = False): ...

traverse_dfs: Incomplete
traverse_bfs: Incomplete

@dataclass
class EventMetrics:
    duration_time_ns: int = ...
    self_time_ns: int = ...
    idle_time_ns: int = ...
    queue_depth: int = ...
    @property
    def fraction_idle_time(self): ...

@dataclass
class Interval:
    start: int
    end: int
    queue_depth: int = ...

class EventKey:
    event: Incomplete
    def __init__(self, event) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...
    def intervals_overlap(self, intervals: list[Interval]): ...

class BasicEvaluation:
    profile: Incomplete
    metrics: dict[EventKey, EventMetrics]
    event_keys: Incomplete
    events: Incomplete
    cuda_events: list[_KinetoEvent]
    queue_depth_list: Incomplete
    def __init__(self, prof: profile) -> None: ...
    def compute_self_time(self) -> None:
        """
        Computes event's self time(total time - time in child ops).
        """
    def compute_queue_depth(self):
        """
        Computes queue_depth at each event. This will calculate the queue depth data for
        All the events in the tree.
        This will return a list of Interval of queue depth data of cuda launch and kernels.
        """
    def compute_idle_time(self) -> None:
        """
        Computes idle time of the profile.
        """
    def rank_events(self, length):
        """
        Filter and Rank the events based on some heuristics:
        1) Events that are in the falling phase of the queue depth.
        2) Events that have a high idle_time, self_time difference.

        Parameters:
            length: The number of events to return.
        """
    def get_optimizable_events(self, length: int = 1, print_enable: bool = True): ...

def index_of_first_match(seq, predicate, start: int = 0, end=None): ...
def argmax(seq, key=..., start: int = 0, end=None): ...
def source_code_location(event): ...
def _init_for_cuda_graphs() -> None: ...
