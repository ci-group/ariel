import abc
import torch
import torch.utils._pytree as pytree
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from torch import Tensor as Tensor
from torch._C import DispatchKey as DispatchKey
from torch._higher_order_ops.utils import HopInstance as HopInstance, HopSchema as HopSchema, _has_gen_schema as _has_gen_schema, call_op as call_op, materialize_callable_in_args as materialize_callable_in_args, unique_graph_id as unique_graph_id
from torch._ops import HigherOrderOperator as HigherOrderOperator, OpOverload as OpOverload, OperatorBase as OperatorBase
from torch._prims_common import clone_preserve_strides as clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, track_tensor_tree as track_tensor_tree
from typing import Any

class SchemaHolder:
    schema: Incomplete
    def __init__(self, schema: torch.FunctionSchema) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self) -> int: ...
    @classmethod
    def from_tree_spec(cls, tree_spec: pytree.TreeSpec): ...

def get_base(tensor): ...

class ViewInfo(ABC, metaclass=abc.ABCMeta):
    base_index: int
    def __init__(self, base_index) -> None: ...
    @abstractmethod
    def regenerate_view(self, bases_list: list[Tensor]): ...

@dataclass
class AsStridedViewInfo(ViewInfo):
    size: Sequence[int | torch.SymInt]
    stride: Sequence[int | torch.SymInt]
    storage_offset: int
    def __init__(self, base_index, size, stride, storage_offset) -> None: ...
    def regenerate_view(self, bases_list: list[Tensor]): ...

@dataclass
class SliceViewInfo(ViewInfo):
    dim: int | torch.SymInt
    start: int | torch.SymInt
    end: int | torch.SymInt
    def __init__(self, base_index, dim, start, end) -> None: ...
    def regenerate_view(self, bases_list: list[Tensor]): ...

@dataclass
class AliasViewInfo(ViewInfo):
    def __init__(self, base_index) -> None: ...
    def regenerate_view(self, bases_list: list[Tensor]): ...

@dataclass
class NotView(ViewInfo):
    def __init__(self, base_index) -> None: ...
    def regenerate_view(self, bases_list: list[Tensor]): ...

def is_alias(base, tensor): ...
def try_use_slice(base, tensor): ...
def write_view_information_to_args(mutable_arg_names: list[str], mutable_arg_types: list[torch.Type], kwargs: dict[str, Any], arg_to_base_index: dict[str, Any]):
    """
    This function writes the view information into kwargs. It reads mutable_args from kwargs.
    and uses arg_to_base_index and tensor information to write ViewInfo into kwargs.
    mutable_arg_names: mutable custom operator arg names.
    mutable_arg_types: mutable custom operator arg types.
    kwargs: the original custom operator args.
    arg_to_base_index: maps mutable_arg_name to int | [int] that refers to the base tensor that
                       corresponds to the input tensor
    """
def read_view_information_from_args(mutable_arg_names: list[str], mutable_arg_types: list[torch.Type], kwargs: dict[str, Any], all_bases: list[Tensor]):
    """
    This reads the view information added by `write_view_information_to_args` from kwargs, pop them,
    and returns a dict arg_name -> ViewInfo | [ViewInfo](if the input is list). that maps each mutable arg
    to its view information.
    mutable_arg_names: mutable custom operator arg names.
    mutable_arg_types: mutable custom operator arg types.
    kwargs : args of auto_functionalize(custom_op, kwargs)
    """

class AutoFunctionalized(HigherOrderOperator):
    '''auto_functionalized(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.

    Concretely, it looks at all the arguments that are mutable through
    _mutable_op\'s operator schema, clones those kwargs, runs
    `out = _mutable_op(**kwargs)` with the cloned values, and then returns the
    operator output concatenated with the cloned values that were mutated.

    We have some restrictions on `_mutable_op`.
    See `can_auto_functionalize` for the restrictions. We can likely lift
    many of these if users request it.

    The reason why _mutable_op is prefixed with an
    underscore is to prevent collisions with kwarg names in **kwargs.
    '''
    def __init__(self) -> None: ...
    def __call__(self, /, _mutable_op: OpOverload, **kwargs: Any) -> tuple[Any, tuple[Tensor, ...]]: ...

auto_functionalized: Incomplete
_MutableOpType = OpOverload | HigherOrderOperator

class AutoFunctionalizedV2(HigherOrderOperator):
    '''auto_functionalized_v2(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.
    Unlike AutoFunctionalized, this version is improved to better handle
    view tensors. This version is only used in non export mode.
    '''
    def __init__(self) -> None: ...
    def __call__(self, /, _mutable_op: _MutableOpType, **kwargs: Any) -> tuple[Any, tuple[Tensor, ...]]: ...

auto_functionalized_v2: Incomplete

def can_auto_functionalize(op: OperatorBase | HopInstance) -> bool: ...
def get_mutable_args_from_schema(schema: torch.FunctionSchema) -> tuple[list[str], list[torch.Type]]:
    """
    Returns the list of argument names that get mutated according to the
    schema and their types.
    """
def get_mutable_args(op: OpOverload) -> tuple[list[str], list[torch.Type]]: ...
def do_auto_functionalize(mode: torch._subclasses.functional_tensor.FunctionalTensorMode, op: OpOverload, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Functionalizes a call to op(*args, **kwargs) by emitting a call to
    `outs = auto_functionalized(op, normalized_kwargs)`
    and replacing the mutated (args, kwargs) with the corresponding outputs.

    The normalized_kwargs are just the (args, kwargs), but all in kwarg form.
    This makes handling easier for the auto_functionalized HOP.
    """
def do_auto_functionalize_v2(mode: torch._subclasses.functional_tensor.FunctionalTensorMode, op: OpOverload | HopInstance, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...
def auto_functionalized_dense(_mutable_op: OpOverload, _only_clone_these_tensors: tuple[str, ...] | None = None, **kwargs: Any) -> tuple[Any, tuple[Tensor, ...]]: ...
def auto_functionalized_fake(mode, _mutable_op: OpOverload, **kwargs: Any) -> tuple[Any, tuple[Tensor, ...]]: ...
def auto_functionalized_proxy(mode, _mutable_op: OpOverload, **kwargs: Any) -> tuple[Any, tuple[Tensor, ...]]: ...
@auto_functionalized.py_functionalize_impl
def auto_functionalized_func(ctx, _mutable_op, **kwargs): ...
def auto_functionalized_v2_dense(_mutable_op: _MutableOpType, _only_clone_these_bases: tuple[int, ...] | None = None, **kwargs: Any) -> tuple[Any, tuple[Tensor, ...]]: ...
def _generate_new_op_kwargs_from_bases(schema, kwargs, all_bases, _only_clone_these_bases): ...
def auto_functionalized_v2_fake(mode, _mutable_op: _MutableOpType, **kwargs: dict[str, Any]) -> tuple[Any, tuple[Tensor, ...]]: ...
def auto_functionalized_v2_proxy(mode, _mutable_op: _MutableOpType, **kwargs: Any) -> tuple[Any, tuple[Tensor, ...]]: ...
@auto_functionalized_v2.py_functionalize_impl
def auto_functionalized_v2_func(ctx, _mutable_op, **kwargs): ...
