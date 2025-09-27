import torch
from collections.abc import Sequence
from torch._C import ListType as ListType, TupleType as TupleType
from torch.jit._recursive import wrap_cpp_module as wrap_cpp_module
from typing import Any, Callable, NamedTuple, TypeVar

T = TypeVar('T')
MAX_RAW_TENSOR_SIZE: int

class InflatableArg(NamedTuple):
    """Helper type for bundled inputs.

    'value' is the compressed/deflated input that is stored in the model. Value
    must be of the same type as the argument to the function that it is a deflated
    input for.

    'fmt' is a formatable code string that is executed to inflate the compressed data into
    the appropriate input. It can use 'value' as an input to the format str. It must result
    in a value of the same type as 'value'.

    'fmt_fn' is a formatable function code string that is executed to inflate the compressed
    data into the appropriate input. It must result in a value of the same type as 'value'.
    The function name should be the formatable part of the string.

    Note: Only top level InflatableArgs can be inflated. i.e. you cannot place
    an inflatable arg inside of some other structure. You should instead create
    an inflatable arg such that the fmt code string returns the full structure
    of your input.
    """
    value: Any
    fmt: str = ...
    fmt_fn: str = ...

def bundle_inputs(model: torch.jit.ScriptModule, inputs: Sequence[tuple[Any, ...]] | None | dict[Callable, Sequence[tuple[Any, ...]] | None], info: list[str] | dict[Callable, list[str]] | None = None, *, _receive_inflate_expr: list[str] | None = None) -> torch.jit.ScriptModule:
    """Create and return a copy of the specified model with inputs attached.

    The original model is not mutated or changed in any way.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    If inputs is passed in as a list then the inputs will be bundled for 'forward'.
    If inputs is instead passed in as a map then all the methods specified in the map
    will have their corresponding inputs bundled. Info should match watchever type is
    chosen for the inputs.

    The returned model will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions will also be defined on the returned module:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        Alternatively if only bundling inputs for forward the map can be omitted and a singular list of inputs
        can be provided instead.

        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. Alternatively if only bundling inputs for forward the map can be omitted and
    a singular list of information can be provided instead. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """
def augment_model_with_bundled_inputs(model: torch.jit.ScriptModule, inputs: Sequence[tuple[Any, ...]] | None = None, _receive_inflate_expr: list[str] | None = None, info: list[str] | None = None, skip_size_check: bool = False) -> None:
    """Add bundled sample inputs to a model for the forward function.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_forward`.
        If the user chooses this method inputs should be None

      - `inputs` is a list of inputs of form List[Tuple[Any, ...]]. A list of tuples where the elements
        of each tuple are the args that make up one input.
    """
def augment_many_model_functions_with_bundled_inputs(model: torch.jit.ScriptModule, inputs: dict[Callable, Sequence[tuple[Any, ...]] | None], _receive_inflate_expr: list[str] | None = None, info: dict[Callable, list[str]] | None = None, skip_size_check: bool = False) -> None:
    """Add bundled sample inputs to a model for an arbitrary list of public functions.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions are also defined:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """
def _inflate_expr(arg: T, ref: str, inflate_helper_fn_name: str, skip_size_check: bool = False) -> tuple[T | torch.Tensor, str, str | None]: ...
def _get_bundled_inputs_attributes_and_methods(script_module: torch.jit.ScriptModule) -> tuple[list[str], list[str]]: ...
def _get_inflate_helper_fn_name(arg_idx: int, input_idx: int, function_name: str) -> str: ...
def bundle_randn(*size, dtype=None):
    """Generate a tensor that will be inflated with torch.randn."""
def bundle_large_tensor(t):
    """Wrap a tensor to allow bundling regardless of size."""
