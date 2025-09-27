import inspect
import torch
import typing
from _typeshed import Incomplete
from torch import Tensor as Tensor, device as device, dtype as dtype, types as types
from torch.utils._exposed_in import exposed_in as exposed_in

_TestTensor = torch.Tensor

def infer_schema(prototype_function: typing.Callable, /, *, mutates_args, op_name: str | None = None) -> str:
    '''Parses the schema of a given function with type hints. The schema is inferred from the
    function\'s type hints, and can be used to define a new operator.

    We make the following assumptions:

    * None of the outputs alias any of the inputs or each other.
    * | String type annotations "device, dtype, Tensor, types" without library specification are
      | assumed to be torch.*. Similarly, string type annotations "Optional, List, Sequence, Union"
      | without library specification are assumed to be typing.*.
    * | Only the args listed in ``mutates_args`` are being mutated. If ``mutates_args`` is "unknown",
      | it assumes that all inputs to the operator are being mutates.

    Callers (e.g. the custom ops API) are responsible for checking these assumptions.

    Args:
        prototype_function: The function from which to infer a schema for from its type annotations.
        op_name (Optional[str]): The name of the operator in the schema. If ``name`` is None, then the
            name is not included in the inferred schema. Note that the input schema to
            ``torch.library.Library.define`` requires a operator name.
        mutates_args ("unknown" | Iterable[str]): The arguments that are mutated in the function.

    Returns:
        The inferred schema.

    Example:
        >>> def foo_impl(x: torch.Tensor) -> torch.Tensor:
        >>>     return x.sin()
        >>>
        >>> infer_schema(foo_impl, op_name="foo", mutates_args={})
        foo(Tensor x) -> Tensor
        >>>
        >>> infer_schema(foo_impl, mutates_args={})
        (Tensor x) -> Tensor
    '''
def derived_types(base_type: type | typing._SpecialForm, cpp_type: str, list_base: bool, optional_base_list: bool, optional_list_base: bool): ...
def get_supported_param_types(): ...

SUPPORTED_RETURN_TYPES: Incomplete

def parse_return(annotation, error_fn): ...

SUPPORTED_PARAM_TYPES: Incomplete

def supported_param(param: inspect.Parameter) -> bool: ...
def tuple_to_list(tuple_type: type[tuple]) -> type[list]:
    """
    Convert `tuple_type` into a list type with the same type arguments. Assumes that `tuple_type` is typing.Tuple type.
    """
