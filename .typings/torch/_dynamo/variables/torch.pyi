import functools
from .. import config as config, graph_break_hints as graph_break_hints, polyfills as polyfills, variables as variables
from ..codegen import PyCodegen as PyCodegen
from ..create_parameter_op import can_convert_to_tracable_parameter as can_convert_to_tracable_parameter, new_parameter_placeholder as new_parameter_placeholder, tracable_create_parameter as tracable_create_parameter
from ..device_interface import get_registered_device_interfaces as get_registered_device_interfaces
from ..exc import unimplemented as unimplemented, unimplemented_v2 as unimplemented_v2
from ..guards import GuardBuilder as GuardBuilder, install_guard as install_guard
from ..source import CallFunctionNoArgsSource as CallFunctionNoArgsSource, SyntheticLocalSource as SyntheticLocalSource
from ..utils import check_unspec_or_constant_args as check_unspec_or_constant_args, guard_if_dyn as guard_if_dyn, has_torch_function as has_torch_function, hashable as hashable, product as product, proxy_args_kwargs as proxy_args_kwargs, unwrap_if_wrapper as unwrap_if_wrapper
from .base import VariableTracker as VariableTracker, typestr as typestr
from .ctx_manager import AutocastModeVariable as AutocastModeVariable, ProfilerContextVariable as ProfilerContextVariable, TorchFunctionDisableVariable as TorchFunctionDisableVariable
from .dicts import ConstDictVariable as ConstDictVariable
from .distributed import DistributedVariable as DistributedVariable, ProcessGroupVariable as ProcessGroupVariable
from .lists import ListVariable as ListVariable, TupleVariable as TupleVariable
from .torch_function import TensorWithTFOverrideVariable as TensorWithTFOverrideVariable, TorchFunctionModeStackVariable as TorchFunctionModeStackVariable, can_dispatch_torch_function as can_dispatch_torch_function, dispatch_torch_function as dispatch_torch_function
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator
from torch._guards import TracingContext as TracingContext
from torch._logging import warning_once as warning_once
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _fsdp_param_group
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type as is_traceable_wrapper_subclass_type
from typing import Any, Callable

log: Incomplete
supported_ctx_manager_classes: Incomplete
REWRITE_OPS_TO_TENSOR_SIZE_METHOD: Incomplete
constant_fold_functions_need_guards: Incomplete
constant_fold_functions: Incomplete

@functools.cache
def tracing_state_functions() -> dict[Callable[[], Any], bool | None]: ...

bin_ops: Incomplete
dispatch_key_set_functions: Incomplete

@functools.cache
def get_overridable_functions(): ...

class BaseTorchVariable(VariableTracker):
    """common base for all torch.* functions, classes, modules and other things"""
    @classmethod
    def create_with_source(cls, value, source): ...
    value: Incomplete
    def __init__(self, value, **kwargs) -> None: ...
    def reconstruct(self, codegen: PyCodegen): ...
    def as_proxy(self): ...
    def as_python_constant(self): ...
    def call_obj_hasattr(self, tx: InstructionTranslator, name): ...
    def can_constant_fold_through(self): ...

class TorchCtxManagerClassVariable(BaseTorchVariable):
    """Points to a context manager class in torch.* that dynamo has implementations"""
    def __repr__(self) -> str: ...
    @staticmethod
    def is_matching_cls(value): ...
    def call_function(self, tx: InstructionTranslator, args: Sequence[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class TorchInGraphFunctionVariable(BaseTorchVariable):
    """Points to a torch function/method that should be put in FX graph"""
    nonstrict_traceable: Incomplete
    def __init__(self, value, nonstrict_traceable=None, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def get_function(self): ...
    @staticmethod
    @functools.cache
    def _get_handlers():
        """Build a dict from function -> method to handle it so that we are O(1)
        in terms of the number of function with special handling."""
    def call_function(self, tx: InstructionTranslator, args: Sequence[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def _call_ntuple(self, tx: InstructionTranslator, args, kwargs):
        """inline behavior of torch.nn.modules.utils._ntuple"""
    @classmethod
    def call_nn_parameter(cls, tx, data=None, requires_grad: bool = True):
        """A call to torch.nn.Parameter() gets lifted to before the graph"""
    @staticmethod
    def _nn_param_via_prefix_insert(tx: InstructionTranslator, data, requires_grad): ...
    def call_tensor_method(self, tx, args, kwargs): ...
    def is_tensor_method(self): ...
    def torch_function_override_enabled(self, tx, args, kwargs): ...

class DispatchKeySetVariable(BaseTorchVariable):
    """represents torch.DispatchKeySet"""
    @staticmethod
    def create(value, **kwargs): ...
    @classmethod
    def create_with_source(cls, value, source): ...
    def is_constant_fold_method(self, name): ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class FuncTorchInterpreterVariable(BaseTorchVariable):
    """represents torch._functorch.pyfunctorch.FuncTorchInterpreter"""
    @classmethod
    def create_with_source(cls, value, source): ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
