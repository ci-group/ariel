__all__ = ['raises', 'expand_tuples', 'reverse_dict', 'groupby', 'typename']

def raises(err, lamda): ...
def expand_tuples(L):
    """
    >>> expand_tuples([1, (2, 3)])
    [(1, 2), (1, 3)]
    >>> expand_tuples([1, 2])
    [(1, 2)]
    """
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
def groupby(func, seq):
    '''Group a collection by a key function
    >>> names = ["Alice", "Bob", "Charlie", "Dan", "Edith", "Frank"]
    >>> groupby(len, names)  # doctest: +SKIP
    {3: [\'Bob\', \'Dan\'], 5: [\'Alice\', \'Edith\', \'Frank\'], 7: [\'Charlie\']}
    >>> iseven = lambda x: x % 2 == 0
    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}
    See Also:
        ``countby``
    '''
def typename(type):
    """Get the name of `type`.
    Parameters
    ----------
    type : Union[Type, Tuple[Type]]
    Returns
    -------
    str
        The name of `type` or a tuple of the names of the types in `type`.
    Examples
    --------
    >>> typename(int)
    'int'
    >>> typename((int, float))
    '(int, float)'
    """
