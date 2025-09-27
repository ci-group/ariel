import torch._C
from _typeshed import Incomplete
from contextlib import contextmanager
from typing import TypeVar
from typing_extensions import ParamSpec

__all__ = ['enable_python_dispatcher', 'no_python_dispatcher', 'enable_pre_dispatch']

no_python_dispatcher = torch._C._DisablePythonDispatcher
enable_python_dispatcher = torch._C._EnablePythonDispatcher
enable_pre_dispatch = torch._C._EnablePreDispatch
_P = ParamSpec('_P')
_T = TypeVar('_T')

class Lit:
    s: Incomplete
    def __init__(self, s) -> None: ...
    def __repr__(self) -> str: ...
