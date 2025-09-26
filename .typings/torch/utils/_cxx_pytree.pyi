import functools
import types
from _typeshed import Incomplete
from collections.abc import Iterable
from optree import PyTreeSpec as TreeSpec
from torch.utils._pytree import KeyEntry as KeyEntry, is_namedtuple as is_namedtuple, is_namedtuple_class as is_namedtuple_class, is_namedtuple_instance as is_namedtuple_instance, is_structseq as is_structseq, is_structseq_class as is_structseq_class, is_structseq_instance as is_structseq_instance
from typing import Any, Callable, TypeVar, overload

__all__ = ['PyTree', 'Context', 'FlattenFunc', 'UnflattenFunc', 'DumpableContext', 'ToDumpableContextFn', 'FromDumpableContextFn', 'TreeSpec', 'LeafSpec', 'keystr', 'key_get', 'register_pytree_node', 'tree_is_leaf', 'tree_flatten', 'tree_flatten_with_path', 'tree_unflatten', 'tree_iter', 'tree_leaves', 'tree_leaves_with_path', 'tree_structure', 'tree_map', 'tree_map_with_path', 'tree_map_', 'tree_map_only', 'tree_map_only_', 'tree_all', 'tree_any', 'tree_all_only', 'tree_any_only', 'treespec_dumps', 'treespec_loads', 'treespec_pprint', 'is_namedtuple', 'is_namedtuple_class', 'is_namedtuple_instance', 'is_structseq', 'is_structseq_class', 'is_structseq_instance']

T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
R = TypeVar('R')
Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], tuple[list[Any], Context]]
UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]
OpTreeUnflattenFunc = Callable[[Context, Iterable[Any]], PyTree]
DumpableContext = Any
ToDumpableContextFn = Callable[[Context], DumpableContext]
FromDumpableContextFn = Callable[[DumpableContext], Context]
KeyPath = tuple[KeyEntry, ...]
FlattenWithKeysFunc = Callable[[PyTree], tuple[list[tuple[KeyEntry, Any]], Any]]

def register_pytree_node(cls, flatten_fn: FlattenFunc, unflatten_fn: UnflattenFunc, *, serialized_type_name: str | None = None, to_dumpable_context: ToDumpableContextFn | None = None, from_dumpable_context: FromDumpableContextFn | None = None, flatten_with_keys_fn: FlattenWithKeysFunc | None = None) -> None:
    """Register a container-like type as pytree node.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_fn (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_fn``.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.
            The function should return an instance of ``cls``.
        serialized_type_name (str, optional): A keyword argument used to specify the fully
            qualified name used when serializing the tree spec.
        to_dumpable_context (callable, optional): An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable representation. This is
            used for json serialization, which is being used in :mod:`torch.export` right now.
        from_dumpable_context (callable, optional): An optional keyword argument to custom specify
            how to convert the custom json dumpable representation of the context back to the
            original context. This is used for json deserialization, which is being used in
            :mod:`torch.export` right now.

    Example::

        >>> # xdoctest: +SKIP
        >>> # Registry a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda children, _: set(children),
        ... )
    """
def tree_is_leaf(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool:
    """Check if a pytree is a leaf.

    >>> tree_is_leaf(1)
    True
    >>> tree_is_leaf(None)
    True
    >>> tree_is_leaf([1, 2, 3])
    False
    >>> tree_is_leaf((1, 2, 3), is_leaf=lambda x: isinstance(x, tuple))
    True
    >>> tree_is_leaf({'a': 1, 'b': 2, 'c': 3})
    False
    >>> tree_is_leaf({'a': 1, 'b': 2, 'c': None})
    False

    Args:
        tree (pytree): A pytree to check if it is a leaf node.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A boolean indicating if the pytree is a leaf node.
    """
def tree_flatten(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> tuple[list[Any], TreeSpec]:
    '''Flatten a pytree.

    See also :func:`tree_unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> tree_flatten(tree)
    ([2, 3, 4, 1, None, 5], PyTreeSpec({\'b\': (*, [*, *]), \'a\': *, \'c\': *, \'d\': *}, NoneIsLeaf, namespace=\'torch\'))
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*, NoneIsLeaf, namespace=\'torch\'))
    >>> tree_flatten(None)
    ([None], PyTreeSpec(*, NoneIsLeaf, namespace=\'torch\'))
    >>> from collections import OrderedDict
    >>> tree = OrderedDict([("b", (2, [3, 4])), ("a", 1), ("c", None), ("d", 5)])
    >>> tree_flatten(tree)
    ([2, 3, 4, 1, None, 5], PyTreeSpec(OrderedDict({\'b\': (*, [*, *]), \'a\': *, \'c\': *, \'d\': *}), NoneIsLeaf, namespace=\'torch\'))

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    '''
def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    '''Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(leaves, treespec)
    True

    Args:
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.
        treespec (TreeSpec): The treespec to reconstruct.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    '''
def tree_iter(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> Iterable[Any]:
    '''Get an iterator over the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> list(tree_iter(tree))
    [2, 3, 4, 1, None, 5]
    >>> list(tree_iter(1))
    [1]
    >>> list(tree_iter(None))
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        An iterator over the leaf values.
    '''
def tree_leaves(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> list[Any]:
    '''Get the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> tree_leaves(tree)
    [2, 3, 4, 1, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A list of leaf values.
    '''
def tree_structure(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> TreeSpec:
    '''Get the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {"b": (2, [3, 4]), "a": 1, "c": None, "d": 5}
    >>> tree_structure(tree)
    PyTreeSpec({\'b\': (*, [*, *]), \'a\': *, \'c\': *, \'d\': *}, NoneIsLeaf, namespace=\'torch\')
    >>> tree_structure(1)
    PyTreeSpec(*, NoneIsLeaf, namespace=\'torch\')
    >>> tree_structure(None)
    PyTreeSpec(*, NoneIsLeaf, namespace=\'torch\')

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A treespec object representing the structure of the pytree.
    '''
def tree_map(func: Callable[..., Any], tree: PyTree, *rests: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree:
    '''Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {"x": 7, "y": (42, 64)})
    {\'x\': 8, \'y\': (43, 65)}
    >>> tree_map(lambda x: x is None, {"x": 7, "y": (42, 64), "z": None})
    {\'x\': False, \'y\': (False, False), \'z\': True}

    If multiple inputs are given, the structure of the tree is taken from the first input;
    subsequent inputs need only have ``tree`` as a prefix:

    >>> tree_map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(x, *xs)`` where ``x`` is the value at the corresponding leaf in ``tree`` and ``xs``
        is the tuple of values at corresponding nodes in ``rests``.
    '''
def tree_map_(func: Callable[..., Any], tree: PyTree, *rests: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree:
    """Like :func:`tree_map`, but do an inplace call on each leaf and return the original tree.

    See also :func:`tree_map`.

    Args:
        func (callable): A function that takes ``1 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf (callable, optional): An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns:
        The original ``tree`` with the value at each leaf is given by the side-effect of function
        ``func(x, *xs)`` (not the return value) where ``x`` is the value at the corresponding leaf
        in ``tree`` and ``xs`` is the tuple of values at values at corresponding nodes in ``rests``.
    """
Type2 = tuple[type[T], type[S]]
Type3 = tuple[type[T], type[S], type[U]]
TypeAny = type[Any] | tuple[type[Any], ...] | types.UnionType
Fn2 = Callable[[T | S], R]
Fn3 = Callable[[T | S | U], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]
MapOnlyFn = Callable[[T], Callable[[Any], Any]]

@overload
def tree_map_only(type_or_types_or_pred: type[T], /, func: Fn[T, Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only(type_or_types_or_pred: Type2[T, S], /, func: Fn2[T, S, Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only(type_or_types_or_pred: Type3[T, S, U], /, func: Fn3[T, S, U, Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only(type_or_types_or_pred: TypeAny, /, func: FnAny[Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only(type_or_types_or_pred: Callable[[Any], bool], /, func: FnAny[Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only_(type_or_types_or_pred: type[T], /, func: Fn[T, Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only_(type_or_types_or_pred: Type2[T, S], /, func: Fn2[T, S, Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only_(type_or_types_or_pred: Type3[T, S, U], /, func: Fn3[T, S, U, Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only_(type_or_types_or_pred: TypeAny, /, func: FnAny[Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
@overload
def tree_map_only_(type_or_types_or_pred: Callable[[Any], bool], /, func: FnAny[Any], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree: ...
def tree_all(pred: Callable[[Any], bool], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool: ...
def tree_any(pred: Callable[[Any], bool], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool: ...
@overload
def tree_all_only(type_or_types: type[T], /, pred: Fn[T, bool], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool: ...
@overload
def tree_all_only(type_or_types: Type2[T, S], /, pred: Fn2[T, S, bool], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool: ...
@overload
def tree_all_only(type_or_types: Type3[T, S, U], /, pred: Fn3[T, S, U, bool], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool: ...
@overload
def tree_any_only(type_or_types: type[T], /, pred: Fn[T, bool], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool: ...
@overload
def tree_any_only(type_or_types: Type2[T, S], /, pred: Fn2[T, S, bool], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool: ...
@overload
def tree_any_only(type_or_types: Type3[T, S, U], /, pred: Fn3[T, S, U, bool], tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> bool: ...
def treespec_dumps(treespec: TreeSpec, protocol: int | None = None) -> str:
    """Serialize a treespec to a JSON string."""
@functools.lru_cache
def treespec_loads(serialized: str) -> TreeSpec:
    """Deserialize a treespec from a JSON string."""

class _DummyLeaf:
    def __repr__(self) -> str: ...

def treespec_pprint(treespec: TreeSpec) -> str: ...

class LeafSpecMeta(Incomplete):
    def __instancecheck__(self, instance: object) -> bool: ...

class LeafSpec(TreeSpec, metaclass=LeafSpecMeta):
    def __new__(cls) -> LeafSpec: ...

def tree_flatten_with_path(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> tuple[list[tuple[KeyPath, Any]], TreeSpec]:
    """Flattens a pytree like :func:`tree_flatten`, but also returns each leaf's key path.

    Args:
        tree: a pytree to flatten. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A tuple where the first element is a list of (key path, leaf) pairs, and the
        second element is a :class:`TreeSpec` representing the structure of the flattened
        tree.
    """
def tree_leaves_with_path(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> list[tuple[KeyPath, Any]]:
    """Gets the leaves of a pytree like ``tree_leaves`` and returns each leaf's key path.

    Args:
        tree: a pytree. If it contains a custom type, that type must be
            registered with an appropriate `tree_flatten_with_path_fn` when registered
            with :func:`register_pytree_node`.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.
    Returns:
        A list of (key path, leaf) pairs.
    """
def tree_map_with_path(func: Callable[..., Any], tree: PyTree, *rests: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree:
    """Like :func:`tree_map`, but the provided callable takes an additional key path argument.

    Args:
        func: A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees. The first positional argument
            to ``func`` is the key path of the leaf in question. The second
            positional argument is the value of the leaf.
        tree: A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests: A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        is_leaf: An extra leaf predicate function that will be called at each
            flattening step. The function should have a single argument with signature
            ``is_leaf(node) -> bool``. If it returns :data:`True`, the whole subtree being treated
            as a leaf. Otherwise, the default pytree registry will be used to determine a node is a
            leaf or not. If the function is not specified, the default pytree registry will be used.

    Returns
        A new pytree with the same structure as ``tree`` but with the value at each leaf given by
        ``func(keypath, x, *xs)`` where ``keypath`` is the key path at the
        corresponding leaf in ``tree``, ``x`` is the value at that leaf, and
        ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
def keystr(kp: KeyPath) -> str:
    """Given a key path, return a pretty-printed representation."""
def key_get(obj: Any, kp: KeyPath) -> Any:
    """Given an object and a key path, return the value at the key path."""
