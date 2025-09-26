__all__ = ['hashable', 'transitive_get', 'raises', 'reverse_dict', 'xfail', 'freeze']

def hashable(x): ...
def transitive_get(key, d):
    """Transitive dict.get
    >>> d = {1: 2, 2: 3, 3: 4}
    >>> d.get(1)
    2
    >>> transitive_get(1, d)
    4
    """
def raises(err, lamda): ...
def reverse_dict(d):
    '''Reverses direction of dependence dict
    >>> d = {"a": (1, 2), "b": (2, 3), "c": ()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: (\'a\',), 2: (\'a\', \'b\'), 3: (\'b\',)}
    :note: dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.
    '''
def xfail(func) -> None: ...
def freeze(d):
    """Freeze container to hashable form
    >>> freeze(1)
    1
    >>> freeze([1, 2])
    (1, 2)
    >>> freeze({1: 2})  # doctest: +SKIP
    frozenset([(1, 2)])
    """
