import onnx
import onnxscript
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping, Sequence
from onnxscript import evaluator, ir
from torch.onnx._internal.exporter import _errors as _errors, _schemas as _schemas, _tensors as _tensors
from typing import Any

logger: Incomplete
ValidAttributeType: Incomplete
AllowedArgType: Incomplete

def _construct_named_inputs_and_attrs(signature: _schemas.OpSignature, args: Sequence[AllowedArgType], kwargs: Mapping[str, AllowedArgType]) -> tuple[dict[str, AllowedArgType], dict[str, ValidAttributeType]]:
    """Construct two mappings: name to inputs and named to attributes based on the signature and args/kwargs.

    This function uses the OpSignature to determine which argument in args and kwargs corresponds to
    which parameter in the signature. ONNX node inputs are stored in named_inputs, and attributes are
    stored in named_attrs. If an _optional input_ is not provided, it is filled with None.

    Args:
        signature: The OpSignature for the node.
        args: The positional arguments for the node.
        kwargs: The keyword arguments for the node.

    Returns:
        A tuple of two mappings: named_inputs and named_attrs.

    Raises:
        ValueError: If a required parameter is not provided.
    """
def _resolve_parameter_dtypes(signature: _schemas.OpSignature, named_inputs: Mapping[str, AllowedArgType]) -> Mapping[_schemas.TypeConstraintParam, ir.TypeProtocol]:
    """Determine which parameter takes which type.

    Handle non-tensor input corner cases and type promotion.

    Requires:
        All ir.Value in name_inputs should have type set. Their type should be
        compatible with the type_constraint of the corresponding parameter in the signature.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.

    Returns:
        A mapping of Constraint names to ir.TypeProtocol.
    """
def _determine_input_dtype(param: _schemas.Parameter, arg: AllowedArgType, type_binding: Mapping[_schemas.TypeConstraintParam, ir.TypeProtocol]) -> ir.DataType:
    """Determine the dtype of the input that is a mix of Python constants and ir.Value."""
def _allowed_types_are_sequence_types(allowed_types: Iterable[ir.TypeProtocol]) -> bool:
    """Check if all allowed types are Sequence types."""
def _get_or_create_constant(constant_farm: dict[tuple[bool | int | float | str | tuple[int] | tuple[float], ir.DataType], ir.Value], arg: bool | int | float | str | tuple[int] | tuple[float] | tuple[bool] | list[int] | list[float] | list[bool], dtype: ir.DataType, opset: onnxscript.values.Opset) -> ir.Value: ...
def _process_python_constants(signature: _schemas.OpSignature, named_inputs: dict[str, AllowedArgType], type_binding: Mapping[_schemas.TypeConstraintParam, ir.TypeProtocol], constant_farm: dict[tuple[bool | int | float | str | tuple[int] | tuple[float], ir.DataType], ir.Value], opset: onnxscript.values.Opset) -> dict[str, ir.Value | None]:
    """Convert Python constants to Constant nodes and list to Sequence nodes based on the dtype information.

    The added constants will be replacing values in named_inputs in place.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.
        type_binding: A mapping of Constraint names to ir.DataType.
        constant_farm: A dictionary of {(py_value, ir.DataType): ir.Value} to store the deduplicated constants.
        opset: The Opset to use for creating Constant nodes.

    Returns:
        A mapping of parameter names to Python constants converted to constant Nodes.
    """
def _reshape_to_1d_tensor(opset: onnxscript.values.Opset, arg: ir.Value) -> ir.Value:
    """Reshape the input to a 1D tensor."""
def _process_python_sequences(signature: _schemas.OpSignature, named_inputs: dict[str, AllowedArgType], type_binding: Mapping[_schemas.TypeConstraintParam, ir.TypeProtocol], constant_farm: dict[tuple[bool | int | float | str | ir.TensorProtocol | tuple[int] | tuple[float], ir.DataType], ir.Value], opset: onnxscript.values.Opset):
    """Handle three types of sequences.

    1. Variadic inputs
    2. Sequence input of ir.Value,
    3. Sequence of Python constants that contains ir.Value
    """
def _determine_output_number(signature: _schemas.OpSignature, named_attrs: Mapping[str, ValidAttributeType]) -> int:
    """Determine the number of outputs for the node with heuristics."""
def _construct_node(signature: _schemas.OpSignature, named_inputs: Mapping[str, ir.Value | None], named_attrs: Mapping[str, ValidAttributeType], opset: onnxscript.values.Opset, num_outputs: int) -> ir.Node:
    """Construct the node with the inputs and attributes.

    Variadic inputs are flattened.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments. When we
            do not have the schema of an operator, we do not know the names of
            the inputs, in which case the names can be anything because they
            are not used in this function. The data structure is passed in for
            consistency with the other functions.
        named_attrs: The mapping of attribute names to their values.
        num_outputs: The number of outputs for the node.
    """

class OpRecorder(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into ONNX IR."""
    nodes: list[ir.Node]
    opset: Incomplete
    functions: dict[ir.OperatorIdentifier, onnxscript.OnnxFunction | ir.Function]
    constant_farm: Incomplete
    def __init__(self, opset: onnxscript.values.Opset, constant_farm: dict[Any, ir.Value]) -> None: ...
    def _call_op(self, op_signature: _schemas.OpSignature, named_inputs: dict[str, AllowedArgType], named_attrs: dict[str, ValidAttributeType], num_outputs: int) -> Sequence[_tensors.SymbolicTensor]:
        """Record nodes for the given opschema and arguments.

        Args:
            op_signature: The OpSchema containing the node signature.
            named_inputs: The mapping of parameter names to their arguments.
            named_attrs: The mapping of attribute names to their values.
        """
    def eval(self, schema: onnx.defs.OpSchema, args: Sequence[AllowedArgType], kwargs: Mapping[str, AllowedArgType]) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor]: ...
    def eval_function(self, function: onnxscript.OnnxFunction, args: Sequence[AllowedArgType], kwargs: Mapping[str, AllowedArgType]) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor] | bool | int: ...
