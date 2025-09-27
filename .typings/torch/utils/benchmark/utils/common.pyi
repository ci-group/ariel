import contextlib
import dataclasses
from collections.abc import Iterable, Iterator
from typing import Any

__all__ = ['TaskSpec', 'Measurement', 'select_unit', 'unit_to_english', 'trim_sigfig', 'ordered_unique', 'set_torch_threads']

@dataclasses.dataclass(init=True, repr=False, eq=True, frozen=True)
class TaskSpec:
    """Container for information used to define a Timer. (except globals)"""
    stmt: str
    setup: str
    global_setup: str = ...
    label: str | None = ...
    sub_label: str | None = ...
    description: str | None = ...
    env: str | None = ...
    num_threads: int = ...
    @property
    def title(self) -> str:
        """Best effort attempt at a string label for the measurement."""
    def setup_str(self) -> str: ...
    def summarize(self) -> str:
        """Build TaskSpec portion of repr string for other containers."""

@dataclasses.dataclass(init=True, repr=False)
class Measurement:
    """The result of a Timer measurement.

    This class stores one or more measurements of a given statement. It is
    serializable and provides several convenience methods
    (including a detailed __repr__) for downstream consumers.
    """
    number_per_run: int
    raw_times: list[float]
    task_spec: TaskSpec
    metadata: dict[Any, Any] | None = ...
    _sorted_times: tuple[float, ...] = ...
    _warnings: tuple[str, ...] = ...
    _median: float = ...
    _mean: float = ...
    _p25: float = ...
    _p75: float = ...
    def __post_init__(self) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    @property
    def times(self) -> list[float]: ...
    @property
    def median(self) -> float: ...
    @property
    def mean(self) -> float: ...
    @property
    def iqr(self) -> float: ...
    @property
    def significant_figures(self) -> int:
        """Approximate significant figure estimate.

        This property is intended to give a convenient way to estimate the
        precision of a measurement. It only uses the interquartile region to
        estimate statistics to try to mitigate skew from the tails, and
        uses a static z value of 1.645 since it is not expected to be used
        for small values of `n`, so z can approximate `t`.

        The significant figure estimation used in conjunction with the
        `trim_sigfig` method to provide a more human interpretable data
        summary. __repr__ does not use this method; it simply displays raw
        values. Significant figure estimation is intended for `Compare`.
        """
    @property
    def has_warnings(self) -> bool: ...
    def _lazy_init(self) -> None: ...
    def meets_confidence(self, threshold: float = ...) -> bool: ...
    @property
    def title(self) -> str: ...
    @property
    def env(self) -> str: ...
    @property
    def as_row_name(self) -> str: ...
    def __repr__(self) -> str:
        """
        Example repr:
            <utils.common.Measurement object at 0x7f395b6ac110>
              Broadcasting add (4x8)
              Median: 5.73 us
              IQR:    2.25 us (4.01 to 6.26)
              372 measurements, 100 runs per measurement, 1 thread
              WARNING: Interquartile range is 39.4% of the median measurement.
                       This suggests significant environmental influence.
        """
    @staticmethod
    def merge(measurements: Iterable['Measurement']) -> list['Measurement']:
        """Convenience method for merging replicates.

        Merge will extrapolate times to `number_per_run=1` and will not
        transfer any metadata. (Since it might differ between replicates)
        """

def select_unit(t: float) -> tuple[str, float]:
    """Determine how to scale times for O(1) magnitude.

    This utility is used to format numbers for human consumption.
    """
def unit_to_english(u: str) -> str: ...
def trim_sigfig(x: float, n: int) -> float:
    """Trim `x` to `n` significant figures. (e.g. 3.14159, 2 -> 3.10000)"""
def ordered_unique(elements: Iterable[Any]) -> list[Any]: ...
@contextlib.contextmanager
def set_torch_threads(n: int) -> Iterator[None]: ...
