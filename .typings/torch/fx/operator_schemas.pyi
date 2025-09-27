import torch
from .node import Argument
from typing import Any, Callable, NamedTuple

__all__ = ['ArgsKwargsPair', 'check_for_mutable_operation', 'get_signature_for_torch_op', 'create_type_hint', 'type_matches', 'normalize_function', 'normalize_module']

class ArgsKwargsPair(NamedTuple):
    """
    Simple named tuple for wrapping args/kwargs pairs.
    """
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

class _FakeGlobalNamespace:
    def __getattr__(self, name): ...

def check_for_mutable_operation(target: Callable, args: tuple['Argument', ...], kwargs: dict[str, 'Argument']): ...
def get_signature_for_torch_op(op: Callable, return_schemas: bool = False):
    """
    Given an operator on the `torch` namespace, return a list of `inspect.Signature`
    objects corresponding to the overloads of that op.. May return `None` if a signature
    could not be retrieved.

    Args:
        op (Callable): An operator on the `torch` namespace to look up a signature for

    Returns:
        Optional[List[inspect.Signature]]: A list of signatures for the overloads of this
            operator, or None if the operator signatures could not be retrieved. If
            return_schemas=True, returns a tuple containing the optional Python signatures
            and the optional TorchScript Function signature
    """
def create_type_hint(x):
    """
    Produces a type hint for the given argument.

    The :func:`create_type_hint` looks for a type hint compatible with the input argument `x`.

    If `x` is a `list` or `tuple`, it looks for an object in the list whose type is a superclass
    of the rest, and uses that as `base_type` for the `List` or `Tuple` to be returned.
    If no such object is found, it defaults to `List[Any]`.

    If `x` is neither a `list` nor a `tuple`, it returns `x`.
    """
def type_matches(signature_type: Any, argument_type: Any): ...
def normalize_function(target: Callable, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None, arg_types: tuple[Any] | None = None, kwarg_types: dict[str, Any] | None = None, normalize_to_only_use_kwargs: bool = False) -> ArgsKwargsPair | None:
    """
    Returns normalized arguments to PyTorch functions. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs). Does not support modules.

    May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

    Args:
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
        kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
def normalize_module(root: torch.nn.Module, target: str, args: tuple[Any], kwargs: dict[str, Any] | None = None, normalize_to_only_use_kwargs: bool = False) -> ArgsKwargsPair | None:
    """
    Returns normalized arguments to PyTorch modules. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs).

    Args:
        root (nn.Module): root module upon which we query modules
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
