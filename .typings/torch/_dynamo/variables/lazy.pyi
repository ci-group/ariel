from ..utils import is_function_or_wrapper as is_function_or_wrapper
from .base import VariableTracker as VariableTracker
from .tensor import SymNodeVariable as SymNodeVariable
from _typeshed import Incomplete
from typing import Any, Callable
from typing_extensions import Self

class LazyCache:
    """Container to cache the real VariableTracker"""
    value: Incomplete
    source: Incomplete
    vt: VariableTracker | None
    def __init__(self, value: Any, source: Any) -> None: ...
    def realize(self) -> None: ...

class LazyVariableTracker(VariableTracker):
    """
    A structure that defers the creation of the actual VariableTracker
    for a given underlying value until it is accessed.

    The `realize` function invokes VariableTracker.build() to produce the real object.
    Once a LazyVariableTracker has been realized, internal bookkeeping will
    prevent double realization.

    This object should be utilized for processing containers, or objects that
    reference other objects where we may not want to take on creating all the
    VariableTrackers right away.
    """
    _nonvar_fields: Incomplete
    @staticmethod
    def create(value: Any, source: Any, **options: Any) -> LazyVariableTracker: ...
    _cache: Incomplete
    def __init__(self, _cache: LazyCache, **kwargs: Any) -> None: ...
    def realize(self) -> VariableTracker:
        """Force construction of the real VariableTracker"""
    def unwrap(self) -> VariableTracker | Self:
        """Return the real VariableTracker if it already exists"""
    def is_realized(self) -> bool: ...
    def clone(self, **kwargs: Any) -> VariableTracker: ...
    def peek_type(self) -> type[Any]: ...
    def peek_value(self) -> Any: ...
    def __str__(self) -> str: ...
    def __getattr__(self, item: str) -> Any: ...
    visit: Incomplete
    __repr__ = __str__
    @classmethod
    def realize_all(cls, value: Any, cache: dict[int, tuple[Any, Any]] | None = None) -> Any:
        """
        Walk an object and realize all LazyVariableTrackers inside it.
        """
    def is_hashable(self) -> bool: ...
    def original_value(self) -> Any: ...
    def original_source(self) -> Any: ...

class LazySymNodeFormatString:
    sym_node_var: Incomplete
    fmt_var: Incomplete
    def __init__(self, sym_node_variable: SymNodeVariable, fmt_spec_var: VariableTracker) -> None: ...
    def __repr__(self) -> str: ...

def _create_realize_and_forward(name: str) -> Callable[[LazyVariableTracker, Any, Any], Any]: ...
def _populate() -> None: ...
