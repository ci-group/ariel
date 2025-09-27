import enum
from _typeshed import Incomplete
from torch.utils.benchmark.utils import common

__all__ = ['Colorize', 'Compare']

class Colorize(enum.Enum):
    NONE = 'none'
    COLUMNWISE = 'columnwise'
    ROWWISE = 'rowwise'

class _Column:
    _grouped_results: Incomplete
    _flat_results: Incomplete
    _time_scale: Incomplete
    _time_unit: Incomplete
    _trim_significant_figures: Incomplete
    _highlight_warnings: Incomplete
    _template: Incomplete
    def __init__(self, grouped_results: list[tuple[common.Measurement | None, ...]], time_scale: float, time_unit: str, trim_significant_figures: bool, highlight_warnings: bool) -> None: ...
    def get_results_for(self, group): ...
    def num_to_str(self, value: float | None, estimated_sigfigs: int, spread: float | None): ...

class _Row:
    _results: Incomplete
    _row_group: Incomplete
    _render_env: Incomplete
    _env_str_len: Incomplete
    _row_name_str_len: Incomplete
    _time_scale: Incomplete
    _colorize: Incomplete
    _columns: tuple[_Column, ...]
    _num_threads: Incomplete
    def __init__(self, results, row_group, render_env, env_str_len, row_name_str_len, time_scale, colorize, num_threads=None) -> None: ...
    def register_columns(self, columns: tuple[_Column, ...]): ...
    def as_column_strings(self): ...
    @staticmethod
    def color_segment(segment, value, best_value): ...
    def row_separator(self, overall_width): ...
    def finalize_column_strings(self, column_strings, col_widths): ...

class Table:
    results: Incomplete
    _colorize: Incomplete
    _trim_significant_figures: Incomplete
    _highlight_warnings: Incomplete
    label: Incomplete
    row_keys: Incomplete
    column_keys: Incomplete
    def __init__(self, results: list[common.Measurement], colorize: Colorize, trim_significant_figures: bool, highlight_warnings: bool) -> None: ...
    @staticmethod
    def row_fn(m: common.Measurement) -> tuple[int, str | None, str]: ...
    @staticmethod
    def col_fn(m: common.Measurement) -> str | None: ...
    def populate_rows_and_columns(self) -> tuple[tuple[_Row, ...], tuple[_Column, ...]]: ...
    def render(self) -> str: ...

class Compare:
    """Helper class for displaying the results of many measurements in a
    formatted table.

    The table format is based on the information fields provided in
    :class:`torch.utils.benchmark.Timer` (`description`, `label`, `sub_label`,
    `num_threads`, etc).

    The table can be directly printed using :meth:`print` or casted as a `str`.

    For a full tutorial on how to use this class, see:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    Args:
        results: List of Measurment to display.
    """
    _results: list[common.Measurement]
    _trim_significant_figures: bool
    _colorize: Incomplete
    _highlight_warnings: bool
    def __init__(self, results: list[common.Measurement]) -> None: ...
    def __str__(self) -> str: ...
    def extend_results(self, results) -> None:
        """Append results to already stored ones.

        All added results must be instances of ``Measurement``.
        """
    def trim_significant_figures(self) -> None:
        """Enables trimming of significant figures when building the formatted table."""
    def colorize(self, rowwise: bool = False) -> None:
        """Colorize formatted table.

        Colorize columnwise by default.
        """
    def highlight_warnings(self) -> None:
        """Enables warning highlighting when building formatted table."""
    def print(self) -> None:
        """Print formatted table"""
    def _render(self): ...
    def _group_by_label(self, results: list[common.Measurement]): ...
    def _layout(self, results: list[common.Measurement]): ...
