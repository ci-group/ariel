from .dispatch import dispatch as dispatch
from .utils import hashable as hashable
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager

_global_logic_variables: Incomplete
_glv = _global_logic_variables

class Var:
    """Logic Variable"""
    _id: int
    def __new__(cls, *token): ...
    def __str__(self) -> str: ...
    __repr__ = __str__
    def __eq__(self, other): ...
    def __hash__(self): ...

def var(): ...
def vars(): ...
def isvar(v): ...
@contextmanager
def variables(*variables) -> Generator[None]:
    '''
    Context manager for logic variables

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> from __future__ import with_statement
        >>> with variables(1):
        ...     print(isvar(1))
        True
        >>> print(isvar(1))
        False
        >>> # Normal approach
        >>> from unification import unify
        >>> x = var("x")
        >>> unify(x, 1)
        {~x: 1}
        >>> # Context Manager approach
        >>> with variables("x"):
        ...     print(unify("x", 1))
        {\'x\': 1}
    '''
