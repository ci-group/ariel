import sympy
import torch
from .fake_tensor import _DispatchCacheKey as _DispatchCacheKey, _MetadataIntLike as _MetadataIntLike
from dataclasses import dataclass
from torch import SymInt as SymInt
from torch.fx.experimental.sym_node import SymNode as SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv
from torch.types import PySymType as PySymType, py_sym_types as py_sym_types
from torch.utils._backport_slots import dataclass_slots as dataclass_slots

@dataclass(frozen=True)
class _DeconstructedSymNode:
    """
    Represents a SymNode without the associated ShapeEnv
    """
    _expr: sympy.Expr
    pytype: type
    _hint: int | float | bool | None
    constant: int | float | bool | None
    fx_node: torch.fx.Node
    @staticmethod
    def from_node(node: SymNode) -> _DeconstructedSymNode: ...
    def extract(self, shape_env: ShapeEnv) -> SymNode: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def _value_eq(self, other: object) -> bool: ...
    def _value_hash(self) -> int: ...

@dataclass(frozen=True)
class _DeconstructedSymType:
    """
    Represents a SymInt, SymFloat, SymBool without the associated ShapeEnv
    """
    ty: type[PySymType]
    node: _DeconstructedSymNode
    @staticmethod
    def from_sym_type(value: PySymType) -> _DeconstructedSymType: ...
    def extract(self, shape_env: ShapeEnv) -> PySymType: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass(frozen=True)
class _InputBackref:
    value: int

@dataclass
class _PySymInputStub:
    """
    Represents a SymInt in the cached key. Needed because SymInt doesn't
    support __eq__ or __hash__ directly.
    """
    value: PySymType | _DeconstructedSymType | _InputBackref
    def __init__(self, value: PySymType | _DeconstructedSymType | _InputBackref) -> None: ...
    def strip_shape_env(self) -> None: ...
    def extract(self, shape_env: ShapeEnv) -> PySymType: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass
class _SymIntOutputStub:
    """
    Represents a SymInt in the cached output.
    """
    value: int | _DeconstructedSymNode
    def __init__(self, value: SymInt, key_path: int | None) -> None: ...
    def extract(self, key: _DispatchCacheKey, shape_env: ShapeEnv) -> SymInt: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass
class _CacheKeyState:
    """
    State used while building our cache key.
    """
    sym_node_lookup: dict[int, int]
    known_symbols: set[sympy.Symbol]
    shape_env: ShapeEnv | None
    def __init__(self, shape_env: ShapeEnv | None = None) -> None: ...
    def cache_on_shape_env(self) -> bool:
        """
        Returns true if the CacheKey needs to be cached on the ShapeEnv
        rather than the global cache.

        If our inputs contain a SymNode then we can't cache this operation on
        the global cache because the cached output will implicitly depend on
        guard values which might not be true on some other ShapeEnv. So unless
        we're also going to cache the guards we need to cache this operation on
        the ShapeEnv instead of globally.
        """
    def convert_sym_int(self, result: list[object], arg: SymInt) -> None: ...
    def convert_output(self, arg: _MetadataIntLike) -> _MetadataIntLike: ...
