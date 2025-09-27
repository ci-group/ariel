import dataclasses
import torch
from _typeshed import Incomplete
from collections.abc import Iterable, Iterator
from torch import _C as _C, _utils_internal as _utils_internal
from torch._ops import OpOverload as OpOverload
from typing import Any, Callable

def warn_deploy(stacklevel: int = 3) -> None: ...

@dataclasses.dataclass
class Kernel:
    """Models a (function, source location)"""
    func: Callable
    source: str
    def __call__(self, *args, **kwargs): ...

class RegistrationHandle:
    """Does something when someone calls .destroy() on it"""
    _on_destroy: Incomplete
    def __init__(self, on_destroy: Callable) -> None: ...
    def destroy(self) -> None: ...

def get_source(stacklevel: int) -> str:
    '''Get a string that represents the caller.

    Example: "/path/to/foo.py:42"

    Use stacklevel=1 to get the caller\'s source
    Use stacklevel=2 to get the caller\'s caller\'s source
    etc.
    '''
def parse_namespace(qualname: str) -> tuple[str, str]: ...
def lookup_op(qualname: str) -> OpOverload: ...
def is_builtin(op: OpOverload) -> bool: ...
def is_functional_schema(schema: Any) -> bool:
    """Check if the schema is functional.

    An operator is functional if:
    - it does not mutate any of its inputs
    - it does not return a view on any of its inputs
    - it has at least one return
    """
def is_tensorlist_like_type(typ: Any) -> bool: ...
def is_tensor_like_type(typ: Any) -> bool: ...
def mutates_and_returns_first_arg(op: OpOverload):
    """Check if an op is an inplace aten op, i.e. it mutates and returns the first arg.

    TODO: torchgen/model.py's FunctionSchema.parse is the source of truth for this,
    but not all PyTorch builds have torchgen (due to the yaml dependency being weird).
    Figure this out.

    Example: add_(Tensor(a!) x, Tensor y) -> Tensor(a)
    """
def fill_defaults(schema, args, kwargs): ...
def zip_schema(schema: _C.FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Iterable[tuple[_C.Argument, Any]]:
    """zips schema.arguments and (args, kwargs) together.

    Assumes that (args, kwargs) were the inputs to some torch._ops.OpOverload:
    that is, (args, kwargs) must be bindable to the schema (args, kwargs).
    """
def hop_schema_from_fx_node(node): ...
def can_generate_trivial_fake_impl(op: OpOverload) -> bool: ...
def requires_set_python_module() -> bool:
    '''If an op was defined in C++ and extended from Python using the
    torch.library APIs, returns if we require that there have been a
    m.set_python_module("mylib.ops") call from C++ that associates
    the C++ op with a python module.
    '''
def handle_dispatch_mode(curr_mode, op_overload, *args, **kwargs): ...
def has_kwarg_only_args(schema: _C.FunctionSchema): ...
def has_kwarg_only_tensors(schema: _C.FunctionSchema): ...
def has_tensor_arg(schema: _C.FunctionSchema) -> bool:
    """
    Given a schema, returns True if the schema has a Tensor arg.
    A Tensor arg is any arg with a type annotation that might involve Tensor.
    """
def get_device_arg_index(schema: _C.FunctionSchema) -> int | None:
    """
    Given a schema, returns the id of the `device: torch.device` argument.
    If it does not exist, returns None.
    """
def iter_tensors(args: tuple[Any], kwargs: dict[str, Any], allowed_nesting: int = 1) -> Iterator[torch.Tensor]: ...
def check_aliasing_constraint(name, prev, result, get_module=...):
    """
    custom operators' outputs must not alias any inputs or other outputs.
    """
def _c_check_aliasing_constraint(name, args, kwargs, result, get_module=...):
    """
    custom operators' outputs must not have any aliases
    This version uses C++ implementation for perf.
    Only List container is supported.
    Tensors in Lists with not only Tensors are checked.
    """

class MutationChecker:
    """
    Check if an operator mutated its arguments.
    Usage:

    checker = MutationChecker(op, flat_args, args_spec)
    op(*args, **kwargs)
    checker.check()
    """
    op: Incomplete
    args_spec: Incomplete
    flat_args: Incomplete
    real_pre_hashes: Incomplete
    def __init__(self, op, flat_args, args_spec) -> None: ...
    def check(self) -> None: ...

def hash_tensor(t: torch.Tensor) -> torch.Tensor:
    """Some inexpensive hash. Used as a quick and dirty indicator for tensor mutation"""
def has_fake_kernel(op: torch._ops.OpOverload) -> bool:
    """If an operator (that stays alive until FakeTensorMode) has a Fake kernel.
    Don't use this if the operator decomposes before FakeTensorMode.
    """
def mutated_args_kwargs(schema: _C.FunctionSchema) -> tuple[list[int], list[str]]: ...

tags_by_priority: Incomplete

def get_layout_constraint_tag(fn, *, with_default: bool = True): ...
