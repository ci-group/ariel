from typing import Any, Callable, TypeVar

_BACK_COMPAT_OBJECTS: dict[Any, None]
_MARKED_WITH_COMPATIBILITY: dict[Any, None]
_T = TypeVar('_T')

def compatibility(is_backward_compatible: bool) -> Callable[[_T], _T]: ...
