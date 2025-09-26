from _typeshed import Incomplete
from typing import Callable, TypeVar
from typing_extensions import Concatenate, ParamSpec

_P = ParamSpec('_P')
_T = TypeVar('_T')
_C = TypeVar('_C')
_cache_sentinel: Incomplete

def cache_method(f: Callable[Concatenate[_C, _P], _T]) -> Callable[Concatenate[_C, _P], _T]:
    """
    Like `@functools.cache` but for methods.

    `@functools.cache` (and similarly `@functools.lru_cache`) shouldn't be used
    on methods because it caches `self`, keeping it alive
    forever. `@cache_method` ignores `self` so won't keep `self` alive (assuming
    no cycles with `self` in the parameters).

    Footgun warning: This decorator completely ignores self's properties so only
    use it when you know that self is frozen or won't change in a meaningful
    way (such as the wrapped function being pure).
    """
