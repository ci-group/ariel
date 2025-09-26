from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['MDNotImplementedError', 'ambiguity_warn', 'halt_ordering', 'restart_ordering', 'variadic_signature_matches_iter', 'variadic_signature_matches', 'Dispatcher', 'source', 'MethodDispatcher', 'str_signature', 'warning_text']

class MDNotImplementedError(NotImplementedError):
    """A NotImplementedError for multiple dispatch"""

def ambiguity_warn(dispatcher, ambiguities) -> None:
    """Raise warning when ambiguity is detected
    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher
    See Also:
        Dispatcher.add
        warning_text
    """
def halt_ordering() -> None:
    """Deprecated interface to temporarily disable ordering."""
def restart_ordering(on_ambiguity=...) -> None:
    """Deprecated interface to temporarily resume ordering."""
def variadic_signature_matches_iter(types, full_signature) -> Generator[Incomplete]:
    """Check if a set of input types matches a variadic signature.
    Notes
    -----
    The algorithm is as follows:
    Initialize the current signature to the first in the sequence
    For each type in `types`:
        If the current signature is variadic
            If the type matches the signature
                yield True
            Else
                Try to get the next signature
                If no signatures are left we can't possibly have a match
                    so yield False
        Else
            yield True if the type matches the current signature
            Get the next signature
    """
def variadic_signature_matches(types, full_signature): ...

class Dispatcher:
    '''Dispatch methods based on type signature
    Use ``dispatch`` to add implementations
    Examples
    --------
    >>> # xdoctest: +SKIP("bad import name")
    >>> from multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1
    >>> @dispatch(float)
    ... def f(x):
    ...     return x - 1
    >>> f(3)
    4
    >>> f(3.0)
    2.0
    '''
    __slots__: Incomplete
    name: Incomplete
    funcs: Incomplete
    doc: Incomplete
    _cache: Incomplete
    def __init__(self, name, doc=None) -> None: ...
    def register(self, *types, **kwargs):
        '''register dispatcher with new implementation
        >>> # xdoctest: +SKIP
        >>> f = Dispatcher("f")
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1
        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1
        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]
        >>> f(1)
        2
        >>> f(1.0)
        0.0
        >>> f([1, 2, 3])
        [3, 2, 1]
        '''
    @classmethod
    def get_func_params(cls, func): ...
    @classmethod
    def get_func_annotations(cls, func):
        """get annotations of function positional parameters"""
    def add(self, signature, func) -> None:
        '''Add new types/method pair to dispatcher
        >>> # xdoctest: +SKIP
        >>> D = Dispatcher("add")
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)
        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>
        >>> # When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        >>> # with a dispatcher/itself, and a set of ambiguous type signature pairs
        >>> # as inputs.  See ``ambiguity_warn`` for an example.
        '''
    @property
    def ordering(self): ...
    _ordering: Incomplete
    def reorder(self, on_ambiguity=...): ...
    def __call__(self, *args, **kwargs): ...
    def __str__(self) -> str: ...
    __repr__ = __str__
    def dispatch(self, *types):
        """Determine appropriate implementation for this type signature
        This method is internal.  Users should call this object as a function.
        Implementation resolution occurs within the ``__call__`` method.
        >>> # xdoctest: +SKIP
        >>> from multipledispatch import dispatch
        >>> @dispatch(int)
        ... def inc(x):
        ...     return x + 1
        >>> implementation = inc.dispatch(int)
        >>> implementation(3)
        4
        >>> print(inc.dispatch(float))
        None
        See Also:
          ``multipledispatch.conflict`` - module to determine resolution order
        """
    def dispatch_iter(self, *types) -> Generator[Incomplete]: ...
    def resolve(self, types):
        """Determine appropriate implementation for this type signature
        .. deprecated:: 0.4.4
            Use ``dispatch(*types)`` instead
        """
    def __getstate__(self): ...
    def __setstate__(self, d) -> None: ...
    @property
    def __doc__(self): ...
    def _help(self, *args): ...
    def help(self, *args, **kwargs) -> None:
        """Print docstring for the function corresponding to inputs"""
    def _source(self, *args): ...
    def source(self, *args, **kwargs) -> None:
        """Print source code for the function corresponding to inputs"""

def source(func): ...

class MethodDispatcher(Dispatcher):
    """Dispatch methods based on type signature
    See Also:
        Dispatcher
    """
    __slots__: Incomplete
    @classmethod
    def get_func_params(cls, func): ...
    obj: Incomplete
    cls: Incomplete
    def __get__(self, instance, owner): ...
    def __call__(self, *args, **kwargs): ...

def str_signature(sig):
    """String representation of type signature
    >>> str_signature((int, float))
    'int, float'
    """
def warning_text(name, amb):
    """The text for ambiguity warnings"""
