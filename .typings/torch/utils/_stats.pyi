from collections import OrderedDict
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

simple_call_counter: OrderedDict[str, int]
_P = ParamSpec('_P')
_R = TypeVar('_R')

def count_label(label: str) -> None: ...
def count(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
