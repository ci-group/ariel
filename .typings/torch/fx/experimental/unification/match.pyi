from .core import reify as reify, unify as unify
from .unification_tools import first as first, groupby as groupby
from .utils import _toposort as _toposort, freeze as freeze
from .variable import isvar as isvar
from _typeshed import Incomplete

class Dispatcher:
    name: Incomplete
    funcs: Incomplete
    ordering: Incomplete
    def __init__(self, name) -> None: ...
    def add(self, signature, func) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def resolve(self, args): ...
    def register(self, *signature): ...

class VarDispatcher(Dispatcher):
    '''A dispatcher that calls functions with variable names
    >>> # xdoctest: +SKIP
    >>> d = VarDispatcher("d")
    >>> x = var("x")
    >>> @d.register("inc", x)
    ... def f(x):
    ...     return x + 1
    >>> @d.register("double", x)
    ... def f(x):
    ...     return x * 2
    >>> d("inc", 10)
    11
    >>> d("double", 10)
    20
    '''
    def __call__(self, *args, **kwargs): ...

global_namespace: Incomplete

def match(*signature, **kwargs): ...
def supercedes(a, b):
    """``a`` is a more specific match than ``b``"""
def edge(a, b, tie_breaker=...):
    """A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
def ordering(signatures):
    """A sane ordering of signatures to check, first to last
    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
