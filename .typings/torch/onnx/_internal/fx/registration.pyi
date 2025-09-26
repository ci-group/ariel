import dataclasses
import onnxscript
import torch._ops
import types

@dataclasses.dataclass(frozen=True, eq=True)
class ONNXFunction:
    """A wrapper of onnx-script function.

    op_full_name: The qualified name of the function. In the form of '<namespace>::<op_name>.<overload>'.
    onnx_function: The onnx-script function from torchlib.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.

    """
    onnx_function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction
    op_full_name: str
    is_custom: bool = ...
    is_complex: bool = ...

@dataclasses.dataclass(frozen=True, eq=True)
class OpName:
    """A class representing an operator name in internal ONNX converter."""
    namespace: str
    op_name: str
    overload: str
    @classmethod
    def from_name_parts(cls, namespace: str, op_name: str, overload: str | None = None) -> OpName: ...
    @classmethod
    def from_qualified_name(cls, qualified_name: str) -> OpName:
        """When the name is <namespace>::<op_name>[.<overload>]"""
    @classmethod
    def from_op_overload(cls, op_overload: torch._ops.OpOverload) -> OpName: ...
    @classmethod
    def from_builtin_function(cls, builtin_function: types.BuiltinFunctionType) -> OpName:
        """From a builtin function, e.g. operator.add, math.ceil, etc, get the OpName.

        FX graph uses built-in functions to caculate sympy expression. This function
        is used to get the OpName from a builtin function.

        Args:
            builtin_function (types.BuiltinFunctionType): operator.add, math.ceil, etc.

        Returns:
            OpName: _description_
        """
    def qualified_name(self) -> str: ...
