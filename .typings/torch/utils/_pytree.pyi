import dataclasses
import functools
import json
import types
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Callable, ClassVar, Final, Generic, NoReturn, Protocol, TypeVar, overload
from typing_extensions import NamedTuple, Self

__all__ = ['PyTree', 'Context', 'FlattenFunc', 'UnflattenFunc', 'DumpableContext', 'ToDumpableContextFn', 'FromDumpableContextFn', 'TreeSpec', 'LeafSpec', 'keystr', 'key_get', 'register_pytree_node', 'tree_is_leaf', 'tree_flatten', 'tree_flatten_with_path', 'tree_unflatten', 'tree_iter', 'tree_leaves', 'tree_leaves_with_path', 'tree_structure', 'tree_map', 'tree_map_with_path', 'tree_map_', 'tree_map_only', 'tree_map_only_', 'tree_all', 'tree_any', 'tree_all_only', 'tree_any_only', 'treespec_dumps', 'treespec_loads', 'treespec_pprint', 'is_namedtuple', 'is_namedtuple_class', 'is_namedtuple_instance', 'is_structseq', 'is_structseq_class', 'is_structseq_instance']

T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
R = TypeVar('R')

class KeyEntry(Protocol):
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
    def get(self, parent: Any) -> Any: ...

class EnumEncoder(json.JSONEncoder):
    def default(self, obj: object) -> str | dict[str, Any]: ...
Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], tuple[list[Any], Context]]
UnflattenFunc = Callable[[Iterable[Any], Context], PyTree]
DumpableContext = Any
ToDumpableContextFn = Callable[[Context], DumpableContext]
FromDumpableContextFn = Callable[[DumpableContext], Context]
MaybeFromStrFunc = Callable[[str], tuple[Any, Context, str] | None]
KeyPath = tuple[KeyEntry, ...]
FlattenWithKeysFunc = Callable[[PyTree], tuple[list[tuple[KeyEntry, Any]], Any]]

class NodeDef(NamedTuple):
    type: type[Any]
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc
    flatten_with_keys_fn: FlattenWithKeysFunc | None

class _SerializeNodeDef(NamedTuple):
    typ: type[Any]
    serialized_type_name: str
    to_dumpable_context: ToDumpableContextFn | None
    from_dumpable_context: FromDumpableContextFn | None

def register_pytree_node(cls, flatten_fn: FlattenFunc, unflatten_fn: UnflattenFunc, *, serialized_type_name: str | None = None, to_dumpable_context: ToDumpableContextFn | None = None, from_dumpable_context: FromDumpableContextFn | None = None, flatten_with_keys_fn: FlattenWithKeysFunc | None = None) -> None:
    """Register a container-like type as pytree node.

    Note:
        :func:`register_dataclass` is a simpler way of registering a container-like
        type as a pytree node.

    Args:
        cls: the type to register
        flatten_fn: A callable that takes a pytree and returns a flattened
            representation of the pytree and additional context to represent the
            flattened pytree.
        unflatten_fn: A callable that takes a flattened version of the pytree,
            additional context, and returns an unflattened pytree.
        serialized_type_name: A keyword argument used to specify the fully qualified
            name used when serializing the tree spec.
        to_dumpable_context: An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable
            representation. This is used for json serialization, which is being
            used in torch.export right now.
        from_dumpable_context: An optional keyword argument to custom specify how
            to convert the custom json dumpable representation of the context
            back to the original context. This is used for json deserialization,
            which is being used in torch.export right now.
        flatten_with_keys_fn: An optional keyword argument to specify how to
            access each pytree leaf's keypath when flattening and tree-mapping.
            Like ``flatten_fn``, but in place of a List[leaf], it should return
            a List[(keypath, leaf)].
    """

@dataclasses.dataclass(frozen=True)
class ConstantNode:
    value: Any

@dataclasses.dataclass(frozen=True)
class SequenceKey(Generic[T]):
    idx: int
    def __str__(self) -> str: ...
    def get(self, sequence: Sequence[T]) -> T: ...
K = TypeVar('K', bound=Hashable)

@dataclasses.dataclass(frozen=True)
class MappingKey(Generic[K, T]):
    key: K
    def __str__(self) -> str: ...
    def get(self, mapping: Mapping[K, T]) -> T: ...

@dataclasses.dataclass(frozen=True)
class GetAttrKey:
    name: str
    def __str__(self) -> str: ...
    def get(self, obj: Any) -> Any: ...

def is_namedtuple(obj: object | type) -> bool:
    """Return whether the object is an instance of namedtuple or a subclass of namedtuple."""
def is_namedtuple_class(cls) -> bool:
    """Return whether the class is a subclass of namedtuple."""
def is_namedtuple_instance(obj: object) -> bool:
    """Return whether the object is an instance of namedtuple."""
_T_co = TypeVar('_T_co', covariant=True)

class structseq(tuple[_T_co, ...]):
    """A generic type stub for CPython's ``PyStructSequence`` type."""
    __slots__: ClassVar[tuple[()]]
    n_fields: Final[int]
    n_sequence_fields: Final[int]
    n_unnamed_fields: Final[int]
    def __init_subclass__(cls) -> NoReturn:
        """Prohibit subclassing."""
    def __new__(cls, sequence: Iterable[_T_co], dict: dict[str, Any] = ...) -> Self: ...

def is_structseq(obj: object | type) -> bool:
    """Return whether the object is an instance of PyStructSequence or a class of PyStructSequence."""
def is_structseq_class(cls) -> bool:
    """Return whether the class is a class of PyStructSequence."""
def is_structseq_instance(obj: object) -> bool:
    """Return whether the object is an instance of PyStructSequence."""
_odict_flatten = _ordereddict_flatten
_odict_unflatten = _ordereddict_unflatten

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
    """

@dataclasses.dataclass(init=True, frozen=True, eq=True, repr=False)
class TreeSpec:
    type: Any
    context: Context
    children_specs: list['TreeSpec']
    num_nodes: int = dataclasses.field(init=False)
    num_leaves: int = dataclasses.field(init=False)
    num_children: int = dataclasses.field(init=False)
    def __post_init__(self) -> None: ...
    def __repr__(self, indent: int = 0) -> str: ...
    def __eq__(self, other: PyTree) -> bool: ...
    def is_leaf(self) -> bool: ...
    def flatten_up_to(self, tree: PyTree) -> list[PyTree]: ...
    def unflatten(self, leaves: Iterable[Any]) -> PyTree: ...
    def __hash__(self) -> int: ...

@dataclasses.dataclass(init=True, frozen=True, eq=False, repr=False)
class LeafSpec(TreeSpec):
    type: Any = dataclasses.field(default=None, init=False)
    context: Context = dataclasses.field(default=None, init=False)
    children_specs: list['TreeSpec'] = dataclasses.field(default_factory=list, init=False)
    def __post_init__(self) -> None: ...
    def __repr__(self, indent: int = 0) -> str: ...

def tree_flatten(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> tuple[list[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """
def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
def tree_iter(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> Iterable[Any]:
    """Get an iterator over the leaves of a pytree."""
def tree_leaves(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> list[Any]:
    """Get a list of leaves of a pytree."""
def tree_structure(tree: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> TreeSpec:
    """Get the TreeSpec for a pytree."""
def tree_map(func: Callable[..., Any], tree: PyTree, *rests: PyTree, is_leaf: Callable[[PyTree], bool] | None = None) -> PyTree:
    """Map a multi-input function over pytree args to produce a new pytree.

    See also :func:`tree_map_`.

    >>> tree_map(lambda x: x + 1, {'x': 7, 'y': (42, 64)})
    {'x': 8, 'y': (43, 65)}
    >>> tree_map(lambda x: x is None, {'x': 7, 'y': (42, 64), 'z': None})
    {'x': False, 'y': (False, False), 'z': True}

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
    """
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

@dataclasses.dataclass
class _TreeSpecSchema:
    """
    _TreeSpecSchema is the schema used to serialize the TreeSpec
    It contains the following fields:
    - type: A string name of the type. null for the case of a LeafSpec.
    - context: Any format which is json dumpable
    - children_spec: A list of children serialized specs.
    """
    type: str | None
    context: DumpableContext
    children_spec: list['_TreeSpecSchema']

class _ProtocolFn(NamedTuple):
    treespec_to_json: Callable[[TreeSpec], DumpableContext]
    json_to_treespec: Callable[[DumpableContext], TreeSpec]

def treespec_dumps(treespec: TreeSpec, protocol: int | None = None) -> str: ...
@functools.lru_cache
def treespec_loads(serialized: str) -> TreeSpec: ...

class _DummyLeaf:
    def __repr__(self) -> str: ...

def treespec_pprint(treespec: TreeSpec) -> str: ...
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
