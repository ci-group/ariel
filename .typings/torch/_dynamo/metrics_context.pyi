from _typeshed import Incomplete
from collections.abc import Iterator
from typing import Any, Callable
from typing_extensions import TypeAlias

log: Incomplete

class TopN:
    '''
    Helper to record a list of metrics, keeping only the top N "most expensive" elements.
    '''
    at_most: Incomplete
    heap: list[tuple[int, Any]]
    def __init__(self, at_most: int = 25) -> None: ...
    def add(self, key: Any, val: int) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[tuple[Any, int]]: ...
OnExitType: TypeAlias = Callable[[int, int, dict[str, Any], type[BaseException] | None, BaseException | None], None]

class MetricsContext:
    _on_exit: Incomplete
    _metrics: dict[str, Any]
    _start_time_ns: int
    _level: int
    def __init__(self, on_exit: OnExitType) -> None:
        """
        Use this class as a contextmanager to create a context under which to accumulate
        a set of metrics, e.g., metrics gathered during a compilation. On exit of the
        contextmanager, call the provided 'on_exit' function and pass a dictionary of
        all metrics set during the lifetime of the contextmanager.
        """
    def __enter__(self) -> MetricsContext:
        """
        Initialize metrics recording.
        """
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, _traceback: Any) -> None:
        """
        At exit, call the provided on_exit function.
        """
    def in_progress(self) -> bool:
        """
        True if we've entered the context.
        """
    def increment(self, metric: str, value: int) -> None:
        """
        Increment a metric by a given amount.
        """
    def set(self, metric: str, value: Any, overwrite: bool = False) -> None:
        """
        Set a metric to a given value. Raises if the metric has been assigned previously
        in the current context.
        """
    def set_key_value(self, metric: str, key: str, value: Any) -> None:
        """
        Treats a give metric as a dictionary and set the k and value within it.
        Note that the metric must be a dictionary or not present.

        We allow this to be called multiple times (i.e. for features, it's not uncommon
        for them to be used multiple times within a single compilation).
        """
    def update(self, values: dict[str, Any], overwrite: bool = False) -> None:
        """
        Set multiple metrics directly. This method does NOT increment. Raises if any
        metric has been assigned previously in the current context and overwrite is
        not set to True.
        """
    def update_outer(self, values: dict[str, Any]) -> None:
        """
        Update, but only when at the outermost context.
        """
    def add_to_set(self, metric: str, value: Any) -> None:
        """
        Records a metric as a set() of values.
        """
    def add_top_n(self, metric: str, key: Any, val: int) -> None:
        """
        Records a metric as a TopN set of values.
        """

class RuntimeMetricsContext:
    _on_exit: Incomplete
    _metrics: dict[str, Any]
    _start_time_ns: int
    def __init__(self, on_exit: OnExitType) -> None:
        """
        Similar to MetricsContext, but used to gather the runtime metrics that are
        decoupled from compilation, where there's not a natural place to insert a
        context manager.
        """
    def increment(self, metric: str, value: int, extra: dict[str, Any] | None = None) -> None:
        """
        Increment a metric by a given amount.
        """
    def finish(self) -> None:
        """
        Call the on_exit function with the metrics gathered so far and reset.
        """
