__all__ = ['reify', 'unify']

def reify(e, s):
    """Replace variables of expression with substitution
    >>> # xdoctest: +SKIP
    >>> x, y = var(), var()
    >>> e = (1, x, (3, y))
    >>> s = {x: 2, y: 4}
    >>> reify(e, s)
    (1, 2, (3, 4))
    >>> e = {1: x, 3: (y, 5)}
    >>> reify(e, s)
    {1: 2, 3: (4, 5)}
    """
def unify(u, v, s):
    '''Find substitution so that u == v while satisfying s
    >>> x = var("x")
    >>> unify((1, x), (1, 2), {})
    {~x: 2}
    '''
