from _typeshed import Incomplete
from collections import defaultdict
from collections.abc import Generator
from torch.autograd import DeviceType
from typing import Any, NamedTuple

__all__ = ['EventList', 'FormattedTimesMixin', 'Interval', 'Kernel', 'FunctionEvent', 'FunctionEventAvg', 'StringTable', 'MemRecordsAcc']

class EventList(list):
    """A list of Events (for pretty printing)."""
    _use_device: Incomplete
    _profile_memory: Incomplete
    _tree_built: bool
    _with_flops: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _build_tree(self) -> None: ...
    def __str__(self) -> str: ...
    def _remove_dup_nodes(self) -> None: ...
    def _populate_cpu_children(self):
        """Populate child events into each underlying FunctionEvent object.

        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would have make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partially, this function
        will not record a parent child relationship between then.
        """
    def _set_backward_stacktraces(self): ...
    @property
    def self_cpu_time_total(self): ...
    def table(self, sort_by=None, row_limit: int = 100, max_src_column_width: int = 75, max_name_column_width: int = 55, max_shapes_column_width: int = 80, header=None, top_level_events_only: bool = False):
        """Print an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``xpu_time``,
                ``cpu_time_total``, ``cuda_time_total``, ``xpu_time_total``,
                ``cpu_memory_usage``, ``cuda_memory_usage``, ``xpu_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``,
                ``self_xpu_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda/xpu ops events are omitted for profiler result readability.

        Returns:
            A string containing the table.
        """
    def export_chrome_trace(self, path) -> None:
        """Export an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
    def supported_export_stacks_metrics(self): ...
    def export_stacks(self, path: str, metric: str): ...
    def key_averages(self, group_by_input_shapes: bool = False, group_by_stack_n: int = 0, group_by_overload_name: bool = False):
        """Averages all function events over their keys.

        Args:
            group_by_input_shapes: group entries by
                (event name, input shapes) rather than just event name.
                This is useful to see which input shapes contribute to the runtime
                the most and may help with size-specific optimizations or
                choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

            group_by_overload_name: Differentiate operators by their overload name e.g. aten::add.Tensor
            and aten::add.out will be aggregated separately

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
    def total_average(self):
        """Averages all events.

        Returns:
            A FunctionEventAvg object.
        """

class FormattedTimesMixin:
    """Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    """
    cpu_time_str: Incomplete
    device_time_str: Incomplete
    cpu_time_total_str: Incomplete
    device_time_total_str: Incomplete
    self_cpu_time_total_str: Incomplete
    self_device_time_total_str: Incomplete
    @property
    def cpu_time(self): ...
    @property
    def device_time(self): ...
    @property
    def cuda_time(self): ...

class Interval:
    start: Incomplete
    end: Incomplete
    def __init__(self, start, end) -> None: ...
    def elapsed_us(self):
        """
        Returns the length of the interval
        """

class Kernel(NamedTuple):
    name: Incomplete
    device: Incomplete
    duration: Incomplete

class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""
    id: int
    node_id: int
    name: str
    overload_name: str
    trace_name: str
    time_range: Interval
    thread: int
    fwd_thread: int | None
    kernels: list[Kernel]
    count: int
    cpu_children: list[FunctionEvent]
    cpu_parent: FunctionEvent | None
    input_shapes: tuple[int, ...]
    concrete_inputs: list[Any]
    kwinputs: dict[str, Any]
    stack: list
    scope: int
    use_device: str | None
    cpu_memory_usage: int
    device_memory_usage: int
    is_async: bool
    is_remote: bool
    sequence_nr: int
    device_type: DeviceType
    device_index: int
    device_resource_id: int
    is_legacy: bool
    flops: int | None
    is_user_annotation: bool | None
    self_cpu_percent: int
    total_cpu_percent: int
    total_device_percent: int
    def __init__(self, id, name, thread, start_us, end_us, overload_name=None, fwd_thread=None, input_shapes=None, stack=None, scope: int = 0, use_device=None, cpu_memory_usage: int = 0, device_memory_usage: int = 0, is_async: bool = False, is_remote: bool = False, sequence_nr: int = -1, node_id: int = -1, device_type=..., device_index: int = 0, device_resource_id=None, is_legacy: bool = False, flops=None, trace_name=None, concrete_inputs=None, kwinputs=None, is_user_annotation: bool = False) -> None: ...
    def append_kernel(self, name, device, duration) -> None: ...
    def append_cpu_child(self, child) -> None:
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
    def set_cpu_parent(self, parent) -> None:
        """Set the immediate CPU parent of type FunctionEvent.

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
    @property
    def self_cpu_memory_usage(self): ...
    @property
    def self_device_memory_usage(self): ...
    @property
    def self_cuda_memory_usage(self): ...
    @property
    def cpu_time_total(self): ...
    @property
    def self_cpu_time_total(self): ...
    @property
    def device_time_total(self): ...
    @property
    def cuda_time_total(self): ...
    @property
    def self_device_time_total(self): ...
    @property
    def self_cuda_time_total(self): ...
    @property
    def key(self): ...
    def __repr__(self) -> str: ...

class FunctionEventAvg(FormattedTimesMixin):
    """Used to average stats over multiple FunctionEvent objects."""
    key: str | None
    count: int
    node_id: int
    is_async: bool
    is_remote: bool
    use_device: str | None
    cpu_time_total: int
    device_time_total: int
    self_cpu_time_total: int
    self_device_time_total: int
    input_shapes: list[list[int]] | None
    overload_name: str | None
    stack: list | None
    scope: int | None
    cpu_memory_usage: int
    device_memory_usage: int
    self_cpu_memory_usage: int
    self_device_memory_usage: int
    cpu_children: list[FunctionEvent] | None
    cpu_parent: FunctionEvent | None
    device_type: DeviceType
    is_legacy: bool
    flops: int
    def __init__(self) -> None: ...
    is_user_annotation: Incomplete
    def add(self, other): ...
    def __iadd__(self, other): ...
    def __repr__(self) -> str: ...

class StringTable(defaultdict):
    def __missing__(self, key): ...

class MemRecordsAcc:
    """Acceleration structure for accessing mem_records in interval."""
    _mem_records: Incomplete
    _start_nses: list[int]
    _indices: list[int]
    def __init__(self, mem_records) -> None: ...
    def in_interval(self, start_us, end_us) -> Generator[Incomplete]:
        """
        Return all records in the given interval
        To maintain backward compatibility, convert us to ns in function
        """

# Names in __all__ with no definition:
#   Kernel
