import functools
import torch.nn
from .. import graph_break_hints as graph_break_hints, trace_rules as trace_rules, variables as variables
from ..exc import UnspecializeRestartAnalysis as UnspecializeRestartAnalysis, Unsupported as Unsupported, raise_observed_exception as raise_observed_exception, unimplemented_v2 as unimplemented_v2
from ..guards import GuardBuilder as GuardBuilder, install_guard as install_guard
from ..mutation_guard import GenerationTracker as GenerationTracker
from ..source import AttrSource as AttrSource, ConstDictKeySource as ConstDictKeySource, DictGetItemSource as DictGetItemSource, FSDPNNModuleSource as FSDPNNModuleSource, GetItemSource as GetItemSource, NNModuleSource as NNModuleSource, UnspecializedNNModuleSource as UnspecializedNNModuleSource
from ..utils import get_custom_getattr as get_custom_getattr, get_fake_value as get_fake_value, is_lazy_module as is_lazy_module, is_namedtuple as is_namedtuple, is_safe_constant as is_safe_constant, istensor as istensor, istype as istype, nnmodule_has_hooks as nnmodule_has_hooks, object_has_getattribute as object_has_getattribute, proxy_args_kwargs as proxy_args_kwargs, set_example_value as set_example_value, unpatched_nn_module_call as unpatched_nn_module_call, unpatched_nn_module_call_impl as unpatched_nn_module_call_impl
from .base import ValueMutationNew as ValueMutationNew, VariableTracker as VariableTracker, typestr as typestr
from .functions import invoke_and_store_as_constant as invoke_and_store_as_constant
from .lazy import LazyVariableTracker as LazyVariableTracker
from .lists import SliceVariable as SliceVariable
from .user_defined import UserDefinedObjectVariable as UserDefinedObjectVariable
from _typeshed import Incomplete
from contextlib import contextmanager
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator

def initialize_lazy_module(tx: InstructionTranslator, mod, args, kwargs):
    """
    Fairly coupled helper used by NNModuleVariable and UnspecializedNNModuleVariable.

    Used to cause lazy module to be initialized (and delete its init hook) before tracing. Especially
    useful now that 'allowed' modules graph-break on hooks, calling this first ensures there is no hook
    by the time we trace __call__ and thus no graph-break for lazy allowed modules.
    """
@contextmanager
def record_nn_module_stack(module_key: str, source, tx, mod: torch.nn.Module): ...
def guard_to_detect_forward_monkeypatching(source, mod) -> None: ...

class NNModuleVariable(VariableTracker):
    _nonvar_fields: Incomplete
    module_type: Incomplete
    module_key: Incomplete
    value: Incomplete
    nn_module_stack_source: Incomplete
    def __init__(self, module_type: type, module_key: str, value: torch.nn.Module, **kwargs) -> None: ...
    def get_nn_module_stack_source(self): ...
    def set_nn_module_stack_source(self, source) -> None: ...
    def python_type(self): ...
    def _wrap_submodule(self, tx: InstructionTranslator, source, submod, *key_extra, **options): ...
    def unpack_var_sequence(self, tx): ...
    def call_obj_hasattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...
    def is_training(self, tx): ...
    def convert_to_unspecialized(self, tx) -> None:
        """Restart analysis treating this module as an UnspecializedNNModuleVariable"""
    def has_key_in_generic_dict(self, tx: InstructionTranslator, key): ...
    def _custom_getattr_fallback(self, base, tx, name, obj_source):
        """Check for a __getattr__ and handle it specially if it is implemented"""
    def var_getattr(self, tx: InstructionTranslator, name): ...
    def call_function(self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker], constant: bool = False) -> VariableTracker: ...

class UnspecializedNNModuleVariable(UserDefinedObjectVariable):
    _nonvar_fields: Incomplete
    is_state_mutated: bool
    nn_module_stack_source: Incomplete
    def __init__(self, value, **kwargs) -> None: ...
    def _wrap_source(self, attr_source): ...
    def get_nn_module_stack_source(self): ...
    def set_nn_module_stack_source(self, source) -> None: ...
    @staticmethod
    @functools.cache
    def _nn_module_method_ids(): ...
    def unpack_var_sequence(self, tx): ...
    value_type: Incomplete
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def getattr_helper(self, tx: InstructionTranslator, field, name_vt): ...
    def var_getattr(self, tx: InstructionTranslator, name): ...
    def manually_trace_nn_module_getattr(self, tx: InstructionTranslator, name):
        """
        Dynamo tracing of nn.Module __getattr__ can be expensive if the model
        has deep submodule hierarchy. Since the __getattr__ is stable, we can
        directly look into the underlying datastructures. This saves a lot of
        compilation time.
        """

class UnspecializedBuiltinNNModuleVariable(UnspecializedNNModuleVariable):
    """
    Differentiates between builtin nn modules (e.g. torch.nn.Linear) and user defined nn modules.
    """
    def _wrap_source(self, attr_source): ...

class FSDPManagedNNModuleVariable(UnspecializedNNModuleVariable):
    """
    Tracing behavior: trace into submodules and treat them as Unspecialized, do not
    register parameters to the top-level, treat them as function inputs.

    Guards behavior: if 'skip_fsdp_guards', many guards that would be installed
    by a vanilla UnspecializedNNModuleVariable are simply dropped, on the basis
    that a user wrapping their model in FSDP(model) is already opting into a
    requirement to not modify internal model state, which would already break FSDP without
    compilation.
    """
    source: Incomplete
    def __init__(self, value, **kwargs) -> None: ...
    def _wrap_source(self, attr_source): ...
