import dataclasses
import onnx
from _typeshed import Incomplete
from collections.abc import Iterator, Mapping, Sequence
from onnxscript import ir
from typing import Any

logger: Incomplete

class _Empty:
    def __repr__(self) -> str: ...

_EMPTY_DEFAULT: Incomplete
_PY_TYPE_TO_ATTR_TYPE: Incomplete
_LIST_TYPE_TO_ATTR_TYPE: Incomplete
_ALL_VALUE_TYPES: Incomplete
TypeAnnotationValue = Any

@dataclasses.dataclass(frozen=True)
class TypeConstraintParam:
    '''Type constraint for a parameter.

    Attributes:
        name: Name of the parameter. E.g. "TFloat"
        allowed_types: Allowed types for the parameter.
    '''
    name: str
    allowed_types: set[ir.TypeProtocol]
    description: str = ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    @classmethod
    def any_tensor(cls, name: str, description: str = '') -> TypeConstraintParam: ...
    @classmethod
    def any_value(cls, name: str, description: str = '') -> TypeConstraintParam: ...

@dataclasses.dataclass(frozen=True)
class Parameter:
    """A formal parameter of an operator."""
    name: str
    type_constraint: TypeConstraintParam
    required: bool
    variadic: bool
    default: Any = ...
    def __str__(self) -> str: ...
    def has_default(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class AttributeParameter:
    """A parameter in the function signature that represents an ONNX attribute."""
    name: str
    type: ir.AttributeType
    required: bool
    default: ir.Attr | None = ...
    def __str__(self) -> str: ...
    def has_default(self) -> bool: ...

def _get_type_from_str(type_str: str) -> ir.TensorType | ir.SequenceType | ir.OptionalType:
    '''Converter a type_str from ONNX Opschema to ir.TypeProtocol.

    A type str has the form of "tensor(float)" or composite type like "seq(tensor(float))".
    '''
def _convert_formal_parameter(param: onnx.defs.OpSchema.FormalParameter, type_constraints: Mapping[str, TypeConstraintParam]) -> Parameter:
    """Convert a formal parameter from ONNX Opschema to Parameter."""
def _is_optional(type_: type) -> bool:
    """Returns whether a type_ is an Optional."""
def _get_attr_type(type_: type) -> ir.AttributeType:
    """Obtain the type of the attribute from a Python class."""
def _get_type_constraint_name(type_: TypeAnnotationValue) -> str | None:
    '''Returns the name of the type constraint for a given type annotation.

    Args:
        type_: A Python type.

    Returns:
        The name of the type constraint if it is a TypeVar.
        - Prefixes the name with "Sequence_" if the type annotation is a Sequence[].
    '''
def _get_allowed_types_from_type_annotation(type_: TypeAnnotationValue) -> set[ir.TypeProtocol]:
    """Obtain the allowed types from a type annotation."""

@dataclasses.dataclass
class OpSignature:
    '''Schema for an operator.

    Attributes:
        domain: Domain of the operator. E.g. "".
        name: Name of the operator. E.g. "Add".
        overload: Overload name of the operator.
        params: Input parameters. When the op is an ONNX function definition,
          the order is according to the function signature. This mean we can
          interleave ONNX inputs and ONNX attributes in the list.
        outputs: Output parameters.
    '''
    domain: str
    name: str
    overload: str
    params: Sequence[Parameter | AttributeParameter]
    outputs: Sequence[Parameter]
    params_map: Mapping[str, Parameter | AttributeParameter] = dataclasses.field(init=False, repr=False)
    opset_version: int | None = ...
    def __post_init__(self) -> None: ...
    def get(self, name: str) -> Parameter | AttributeParameter: ...
    def __contains__(self, name: str) -> bool: ...
    def __iter__(self) -> Iterator[Parameter | AttributeParameter]: ...
    def __str__(self) -> str: ...
    @classmethod
    def from_opschema(cls, opschema: onnx.defs.OpSchema) -> OpSignature:
        """Produce an OpSignature from an ONNX Opschema."""
    @classmethod
    def from_function(cls, func, domain: str, name: str | None = None, overload: str = '', *, opset_version: int = 1) -> OpSignature:
        """Produce an OpSignature from a function using type annotation."""
