from _typeshed import Incomplete
from collections.abc import Generator
from torch._C._profiler import _EventType as _EventType, _ExtraFields_PyCCall as _ExtraFields_PyCCall, _ExtraFields_PyCall as _ExtraFields_PyCall, _ExtraFields_TorchOp as _ExtraFields_TorchOp, _ProfilerEvent as _ProfilerEvent
from torch.profiler import profile as profile
from torch.profiler._utils import index_of_first_match as index_of_first_match, traverse_bfs as traverse_bfs, traverse_dfs as traverse_dfs

class Pattern:
    """
    Base class for all patterns, subclass this class and implement match()
    to define custom patterns.

    In subclass, define description and skip property.
    """
    prof: Incomplete
    should_benchmark: Incomplete
    name: str
    description: str
    url: str
    event_tree: Incomplete
    tid_root: dict[int, list[_ProfilerEvent]]
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    @property
    def skip(self): ...
    def report(self, event: _ProfilerEvent): ...
    def eventTreeTraversal(self) -> Generator[Incomplete, Incomplete]:
        """
        Traverse the event tree and yield all events.
        Override this method in subclass to customize the traversal.
        """
    def summary(self, events: list[_ProfilerEvent]): ...
    def benchmark_summary(self, events: list[_ProfilerEvent]): ...
    def match(self, event: _ProfilerEvent):
        """
        Return True if the event matches the pattern.
        This method should be overridden in subclass.
        """
    def matched_events(self): ...
    def root_of(self, event: _ProfilerEvent): ...
    def siblings_of(self, event: _ProfilerEvent): ...
    def next_of(self, event: _ProfilerEvent): ...
    def prev_of(self, event: _ProfilerEvent): ...
    def go_up_until(self, event: _ProfilerEvent, predicate): ...

class NamePattern(Pattern):
    description: Incomplete
    name: Incomplete
    def __init__(self, prof: profile, name: str, should_benchmark: bool = False) -> None: ...
    def match(self, event: _ProfilerEvent): ...

class ExtraCUDACopyPattern(Pattern):
    '''
    This pattern identifies if we creates a constant tensor on CPU and immediately moves it to GPU.
    example: torch.zeros((100, 100)).to("cuda")

    Pattern:
    built-in method                 |built-in method
        ...                         |    aten::to
            aten::fill_/aten::zero_ |        aten::_to_copy

    Algorithm:
    We start at node aten::to, go parent events\' previous events,
    and check if we have a aten::fill_/aten::zero_ as we keep going down the tree.
    We always select the last child in the children list when we go down the tree.
    If at any step we failed, it is not a match.
    '''
    name: str
    description: str
    url: str
    init_ops: Incomplete
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    @property
    def skip(self): ...
    def match(self, event): ...
    def benchmark(self, events: list[_ProfilerEvent]): ...

class ForLoopIndexingPattern(Pattern):
    """
    This pattern identifies if we use a for loop to index a tensor that
    can be vectorized.
    example:
    tensor = torch.empty((100, 100))
    for i in range(100):
        tensor[i] = i

    Pattern:
    aten::select | ... | aten::select | ... (Repeat)

    Algorithm:
    We start at node aten::select, and we check if we can find this alternating patterns.
    We also keep a dictionary to avoid duplicate match in the for loop.
    """
    name: str
    description: str
    visited: set[int]
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    def eventTreeTraversal(self) -> Generator[Incomplete, Incomplete]:
        """
        We need to use BFS traversal order to avoid duplicate match.
        """
    def match(self, event: _ProfilerEvent): ...

class FP32MatMulPattern(Pattern):
    name: str
    description: str
    url: str
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    @property
    def skip(self): ...
    def match(self, event: _ProfilerEvent): ...
    def report(self, event: _ProfilerEvent): ...
    def benchmark(self, events: list[_ProfilerEvent]): ...

class OptimizerSingleTensorPattern(Pattern):
    """
    This pattern identifies if we are using the single-tensor version of an optimizer.
    example:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    By adding foreach=True to enable multi-tensor optimizer, we can gain speedup when
    the kernels are relatively small.

    Pattern:
    XXXXX: _single_tenser_<OPTIMIZER_NAME>

    Algorithm:
    String match
    """
    name: str
    optimizers_with_foreach: Incomplete
    description: str
    url: str
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    def match(self, event: _ProfilerEvent): ...

class SynchronizedDataLoaderPattern(Pattern):
    """
    This pattern identifies if we are using num_workers=0 in DataLoader.
    example:
    torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    Add num_workers=N to the arguments. N depends on system configuration.

    Pattern:
    dataloader.py(...): __iter__
        dataloader.py(...): _get_iterator
            NOT dataloader.py(...): check_worker_number_rationality

    Algorithm:
    If we don't see check_worker_number_rationality call in the dataloader __iter__,
    It is not an asynchronous dataloader.

    """
    name: str
    description: str
    url: str
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    def match(self, event: _ProfilerEvent): ...

class GradNotSetToNonePattern(Pattern):
    """
    This pattern identifies if we are not setting grad to None in zero_grad.
    example:
    optimizer.zero_grad()
    By setting set_to_none=True, we can gain speedup

    Pattern:
    XXXXX: _zero_grad
        NOT aten::zeros
            aten::zero_

    aten::zero_ is called on each parameter in the model.
    We also want to make sure it is not called by aten::zeros.

    Algorithm:
    String match
    """
    name: str
    description: str
    url: str
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    def match(self, event: _ProfilerEvent): ...

class Conv2dBiasFollowedByBatchNorm2dPattern(Pattern):
    """
    This pattern identifies if we are enabling bias in Conv2d which is followed by BatchNorm2d.
    Bias doesn't do anything when followed by batchnorm.
    Pattern:
    nn.Module: Conv2d            | nn.Module: BatchNorm2d
        ...
            aten::conv2d AND dtype of third argument is not null
    The third argument is the bias
    Algorithm:
    String match
    """
    name: str
    description: str
    url: str
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    @property
    def skip(self): ...
    def match(self, event: _ProfilerEvent): ...

class MatMulDimInFP16Pattern(Pattern):
    name: str
    description: str
    url: str
    def __init__(self, prof: profile, should_benchmark: bool = False) -> None: ...
    @property
    def skip(self): ...
    def match(self, event: _ProfilerEvent): ...
    def benchmark(self, events: list[_ProfilerEvent]): ...

def source_code_location(event: _ProfilerEvent | None): ...
def input_shapes(event: _ProfilerEvent): ...
def input_dtypes(event: _ProfilerEvent): ...
def report_all_anti_patterns(prof, should_benchmark: bool = False, print_enable: bool = True, json_report_dir: str | None = None): ...
