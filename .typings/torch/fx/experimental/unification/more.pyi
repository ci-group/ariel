from .core import reify as reify, unify as unify
from .dispatch import dispatch as dispatch

def unifiable(cls):
    '''Register standard unify and reify operations on class
    This uses the type and __dict__ or __slots__ attributes to define the
    nature of the term
    See Also:
    >>> # xdoctest: +SKIP
    >>> class A(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> unifiable(A)
    <class \'unification.more.A\'>
    >>> x = var("x")
    >>> a = A(1, 2)
    >>> b = A(1, x)
    >>> unify(a, b, {})
    {~x: 2}
    '''
def reify_object(o, s):
    '''Reify a Python object with a substitution
    >>> # xdoctest: +SKIP
    >>> class Foo(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...
    ...     def __str__(self):
    ...         return "Foo(%s, %s)" % (str(self.a), str(self.b))
    >>> x = var("x")
    >>> f = Foo(1, x)
    >>> print(f)
    Foo(1, ~x)
    >>> print(reify_object(f, {x: 2}))
    Foo(1, 2)
    '''
def _reify_object_dict(o, s): ...
def _reify_object_slots(o, s): ...
def _reify(o, s):
    """Reify a Python ``slice`` object"""
def unify_object(u, v, s):
    '''Unify two Python objects
    Unifies their type and ``__dict__`` attributes
    >>> # xdoctest: +SKIP
    >>> class Foo(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...
    ...     def __str__(self):
    ...         return "Foo(%s, %s)" % (str(self.a), str(self.b))
    >>> x = var("x")
    >>> f = Foo(1, x)
    >>> g = Foo(1, 2)
    >>> unify_object(f, g, {})
    {~x: 2}
    '''
def _unify(u, v, s):
    """Unify a Python ``slice`` object"""
