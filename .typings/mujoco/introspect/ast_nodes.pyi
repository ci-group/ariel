import dataclasses
from _typeshed import Incomplete
from typing import Sequence

VALID_TYPE_NAME_PATTERN: Incomplete
C_INVALID_TYPE_NAMES: Incomplete

def _is_valid_integral_type(type_str: str):
    """Checks if a string is a valid integral type."""

@dataclasses.dataclass
class ValueType:
    """Represents a C type that is neither a pointer type nor an array type."""
    name: str
    is_const: bool = ...
    is_volatile: bool = ...
    def __init__(self, name: str, is_const: bool = False, is_volatile: bool = False) -> None: ...
    def decl(self, name_or_decl: str | None = None) -> str: ...
    def __str__(self) -> str: ...

@dataclasses.dataclass
class ArrayType:
    """Represents a C array type."""
    inner_type: ValueType | PointerType
    extents: tuple[int, ...]
    def __init__(self, inner_type: ValueType | PointerType, extents: Sequence[int]) -> None: ...
    @property
    def _extents_str(self) -> str: ...
    def decl(self, name_or_decl: str | None = None) -> str: ...
    def __str__(self) -> str: ...

@dataclasses.dataclass
class PointerType:
    """Represents a C pointer type."""
    inner_type: ValueType | ArrayType | PointerType
    is_const: bool = ...
    is_volatile: bool = ...
    is_restrict: bool = ...
    def decl(self, name_or_decl: str | None = None) -> str:
        """Creates a string that declares an object of this type."""
    def __str__(self) -> str: ...

@dataclasses.dataclass
class FunctionParameterDecl:
    """Represents a parameter in a function declaration.

  Note that according to the C language rule, a function parameter of array
  type undergoes array-to-pointer decay, and therefore appears as a pointer
  parameter in an actual C AST. We retain the arrayness of a parameter here
  since the array's extents are informative.
  """
    name: str
    type: ValueType | ArrayType | PointerType
    def __str__(self) -> str: ...
    @property
    def decltype(self) -> str: ...

@dataclasses.dataclass
class FunctionDecl:
    """Represents a function declaration."""
    name: str
    return_type: ValueType | ArrayType | PointerType
    parameters: tuple[FunctionParameterDecl, ...]
    doc: str
    def __init__(self, name: str, return_type: ValueType | ArrayType | PointerType, parameters: Sequence[FunctionParameterDecl], doc: str) -> None: ...
    def __str__(self) -> str: ...
    @property
    def decltype(self) -> str: ...

class _EnumDeclValues(dict[str, int]):
    """A dict with modified stringified representation.

  The __repr__ method of this class adds a trailing comma to the list of values.
  This is done as a hint for code formatters to place one item per line when
  the stringified OrderedDict is used in generated Python code.
  """
    def __repr__(self) -> str: ...

@dataclasses.dataclass
class EnumDecl:
    """Represents an enum declaration."""
    name: str
    declname: str
    values: dict[str, int]
    def __init__(self, name: str, declname: str, values: dict[str, int]) -> None: ...

@dataclasses.dataclass
class StructFieldDecl:
    """Represents a field in a struct or union declaration."""
    name: str
    type: ValueType | ArrayType | PointerType | AnonymousStructDecl | AnonymousUnionDecl
    doc: str
    array_extent: tuple[str | int, ...] | None = ...
    def __str__(self) -> str: ...
    @property
    def decltype(self) -> str: ...

@dataclasses.dataclass
class AnonymousStructDecl:
    """Represents an anonymous struct declaration."""
    fields: tuple[StructFieldDecl | AnonymousUnionDecl, ...]
    def __init__(self, fields: Sequence[StructFieldDecl]) -> None: ...
    def __str__(self) -> str: ...
    def _inner_decl(self): ...
    def decl(self, name_or_decl: str | None = None): ...

class AnonymousUnionDecl(AnonymousStructDecl):
    """Represents an anonymous union declaration."""
    def decl(self, name_or_decl: str | None = None): ...

@dataclasses.dataclass
class StructDecl:
    """Represents a struct declaration."""
    name: str
    declname: str
    fields: tuple[StructFieldDecl | AnonymousUnionDecl, ...]
    def __init__(self, name: str, declname: str, fields: Sequence[StructFieldDecl | AnonymousUnionDecl]) -> None: ...
    def decl(self, name_or_decl: str | None = None) -> str: ...
