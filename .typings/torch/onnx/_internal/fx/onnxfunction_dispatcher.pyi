import onnxscript
import torch.fx
from _typeshed import Incomplete
from collections.abc import Sequence
from onnxscript.function_libs.torch_lib import graph_building as onnxscript_graph_building
from torch.onnx._internal._exporter_legacy import OnnxRegistry as OnnxRegistry
from torch.onnx._internal.fx import registration as registration, type_utils as fx_type_utils
from typing import Any

logger: Incomplete

class OnnxFunctionDispatcher:
    """A dispatcher that finds the best ONNX Function for ATen/Custom operators.

    It uses the `torch.ops` name to find the function. If not found, it falls back to default.
    Otherwise, the best match is found among all function overloads. An exact match has
    higher precedence over the closest ones.

    Below is a breakdown on how the dispatch mechanism works:

    1. Use the torch.ops name to find the function:
        a. Check if the ATen overload exists in the registry.
        b. If not, check if the default overload exists in the registry.

    2. Find the nearest match among all overloaded functions:
        a. If the types match perfectly, select the function.
        b. Otherwise, find the nearest one with the highest matching score. Because of
            the potential wrongly annotated dtypes and attributes matching, we use
            nearest match to find the best function once the aten name is targeted.

    3. Tie-breaker: If there are multiple nearest matches, we will select the one with
        the highest matching score.

    NOTE: The nearest match `doesn't guarantee` a correct match, and a warning message is logged.
    """
    onnx_registry: Incomplete
    def __init__(self, onnx_registry: OnnxRegistry) -> None:
        """Initialize the ONNX Function dispatcher.

        Args:
            onnx_registry: The ONNX registry.
        """
    def dispatch(self, node: torch.fx.Node, onnx_args: Sequence[fx_type_utils.TensorLike | str | int | float | bool | list | complex | None], onnx_kwargs: dict[str, fx_type_utils.Argument]) -> onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction:
        """Dispatches an ONNX function based on the given FX node, arguments, and keyword arguments.
        Args:
            node: The TorchFX node to dispatch the function for.
            onnx_args: The arguments of the ONNX function.
            onnx_kwargs: The keyword arguments of the ONNX function.

        Returns:
            Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
        Raises:
            RuntimeError: If there are no overloaded functions available for the given FX node.
        """
    def _filter_or_keep_complex(self, node, default_and_custom_functions: list[registration.ONNXFunction]) -> list[registration.ONNXFunction]:
        """Filter the complex functions if the input has complex dtype."""
    def _find_the_perfect_or_nearest_match_onnxfunction(self, node: torch.fx.Node, default_and_custom_functions: list[registration.ONNXFunction], onnx_args: Sequence[fx_type_utils.TensorLike | str | int | float | bool | list | complex | None], onnx_kwargs: dict[str, fx_type_utils.Argument]):
        """Find the perfect/nearest matched OnnxFunction for the given FX node, arguments, and keyword arguments.

        Args:
            default_and_custom_functions: The list includes overloaded functions, with
                custom ones appearing after the default ones.
            onnx_args: Arguments organized in PyTorch inputs way.
            onnx_kwargs: Keyword arguments organized in PyTorch inputs way.

            Returns:
                Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
            Raises:
                RuntimeError: If there are no overloaded functions available for the given FX node.
        """
    def _get_aten_name(self, node: torch.fx.Node) -> registration.OpName:
        """Get the OpName from the target.

        Args:
            node: The TorchFX node to get the aten name for.

        Returns:
            The internal op name within dataclass: registration.OpName.
        """
    def get_function_overloads(self, node: torch.fx.Node) -> list[registration.ONNXFunction]:
        """Get the function overloads from the registry.

        Args:
            node: The node to get the function overloads for.

        Returns:
            The list contains ONNXFunctions, starting with the default ones and
            followed by any custom ones.
        """

class _OnnxSchemaChecker:
    '''
    The OnnxSchemaChecker class is a checker for ONNX OpSchema and param schema.

    It provides methods to check for input compatibility based on the OpSchema. It also
    provides a matching score to indicate how well the OpSchema matches the input and
    kwargs types. A function will be evaluated as perfect match, nearest match eligible,
    or no match.

    Here are some common examples in categories:

    1. [NOTE: Perfect match]: The number of inputs and attributes are exactly the same as
        the OpSchema. The types of inputs and attributes are exactly the same as the
        OpSchema.

        ```python
        inputs = (Tensor[2, 3], Tensor[2, 3])
        attributes = {"alpha": 1.0}


        @torch_op("aten::op")
        def aten_op(self: TReal, other: TReal, alpha: float = 1) -> TReal: ...
        ```
        Result: Perfect match.

    2. [NOTE: Optional input]: The dispatcher recognizes optional inputs. However,
        the input can\'t be ignored. None must be provided.

        ```python
        inputs = (Tensor([2, 3]), None)
        attributes = {}

        aten_op(X: TTensor, Y: Optional[INT64]):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::convolution`.

    3. [NOTE: Different attributes]: If an attribute is provided with value, it\'s
        a must to match the attribute in function signature.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a":1, "b":2}

        aten_op(X: TTensor, a: int):
            ...
        ```
        Result: No match.
        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    4. [NOTE: Default attributes]: Default attribute will fill in the value into
        inputs/attributes.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::clone`

    5. [NOTE: Ignore attribute with None value]: The attributes with None value
        will be ignored in matching.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor):
            ...
        ```
        Result: Perfect match.

        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Nearest match eligible.

        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    Attributes:
        onnxfunction: The OnnxFunction.
        param_schema: The parameter schema defined in the OnnxFunction.
        op_schema: The ONNX OpSchema.
        type_constraints: The type constraints defined in the OpSchema.
        attributes: The attributes defined in the OpSchema.
        _matching_score: The matching score of the OnnxSchemaChecker .

    '''
    onnxfunction: Incomplete
    param_schema: Incomplete
    op_schema: Incomplete
    type_constraints: Incomplete
    attributes: Incomplete
    _matching_score: int | None
    def __init__(self, onnxfunction: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction) -> None:
        """Initialize the OnnxSchemaChecker .

        Args:
            onnxfunction: The OnnxFunction.
        """
    @property
    def match_score(self) -> int | None:
        """The matching score of the OnnxSchemaChecker .

        If this remains None, it means the matching score has not been calculated,
        and it's not a nearest match candidate.

        Returns:
            The matching score of the OnnxSchemaChecker .
        """
    def perfect_match_inputs(self, args: Sequence[fx_type_utils.TensorLike | str | int | float | bool | list | complex | None], kwargs: dict[str, fx_type_utils.Argument]) -> bool:
        """Check if the inputs perfectly match the OpSchema requirements.

        The definition of perfect match is that the input types are all in the type
        constraints and the number of inputs matches the number of inputs in the
        OpSchema.

        Checking steps:
        1. The function signature matches the inputs number, and attribute names.
        2. The input/attribute types are all in the type constraints.

        A function should at least pass the first step to be eligible for the
        nearest matching.

        Args:
            args: The input arguments organized in PyTorch inputs way.
            kwargs: The input keyword arguments organized in PyTorch inputs way.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
    def _match_onnx_attribute_type(self, attribute_name: str, attribute: fx_type_utils.Argument | onnxscript_graph_building.TorchScriptTensor, is_sequence: bool = False) -> bool: ...
    def _record_matching_score(self, inputs: Sequence[fx_type_utils.TensorLike | str | int | float | bool | list | complex | None], attributes: dict[str, fx_type_utils.Argument]):
        """Calculate the inputs matching score of the OpSchema requirements to find the nearest match.

        Only the functions which have the same number of inputs and attributes as the
        OpSchema are eligible to be a nearest match candidate. Thus, we don't need to
        check the length of inputs and attributes here, and only check the types of
        inputs and attributes.

        How the matchsing score is calculated:
            score += 1 if one input/attribute type is in the type constraints.

        Limitations:
            None/NoeType/[] could result in zero matches, and the same score of overloads.

        Args:
            inputs: The input arguments.
            attributes: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
    def _separate_input_attributes_from_arguments(self, param_schemas: Sequence[onnxscript.values.ParamSchema], args: Sequence[fx_type_utils.TensorLike | str | int | float | bool | list | complex | None], kwargs: dict[str, fx_type_utils.Argument], fill_defaults: bool = True) -> tuple[list[Any], dict[str, Any]]:
        '''Separate Python args and kwargs into ONNX inputs and attributes.

        Extra_kwargs are ignored if their values are None. For example, if the
        OpSchema has an attribute "rounding_mode" and the caller provides
        "rounding_mode=None", the attribute "rounding_mode" will not be included
        in the returned attributes when the OnnxFunction signature doesn\'t have
        "rounding_mode" as an attribute.

        Args:
            param_schemas: The parameter schemas of an Op or a OnnxFunction.
            args: The Python positional arguments supplied by the caller.
            kwargs: The Python keyword arguments supplied by the caller.
            fill_defaults: Whether to fill the default values for attributes.

        Returns:
            A tuple of two elements:
            - A list of ONNX inputs.
            - An dictionary of ONNX attribute names and values.

        Raises:
            TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
            TypeError: When a required input is not provided.
        '''

def _is_arg_with_complex_dtype(arg: fx_type_utils.Argument) -> bool:
    """Check if the node has complex dtype recursively."""
def _find_onnx_data_type(torch_input: fx_type_utils.TensorLike | str | int | float | bool | list | tuple | complex | None) -> set[str]:
    """Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string."""
