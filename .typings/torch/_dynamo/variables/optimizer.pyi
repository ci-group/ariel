from ..guards import GuardBuilder as GuardBuilder, install_guard as install_guard
from ..source import AttrSource as AttrSource, ConstDictKeySource as ConstDictKeySource, DictGetItemSource as DictGetItemSource, GetItemSource as GetItemSource, GlobalWeakRefSource as GlobalWeakRefSource, GradSource as GradSource
from ..utils import GLOBAL_KEY_PREFIX as GLOBAL_KEY_PREFIX
from .base import VariableTracker as VariableTracker
from .constant import ConstantVariable as ConstantVariable
from .dicts import ConstDictVariable as ConstDictVariable
from .lists import ListVariable as ListVariable
from .misc import GetAttrVariable as GetAttrVariable
from .user_defined import UserDefinedObjectVariable as UserDefinedObjectVariable
from _typeshed import Incomplete
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator
from torch._logging import getArtifactLogger as getArtifactLogger
from torch.utils._pytree import tree_map_only as tree_map_only

class ArgMappingException(Exception): ...
class GuardInstallException(Exception): ...

perf_hint_log: Incomplete

def _is_static_for_cudagraphs(x): ...

class OptimizerVariable(UserDefinedObjectVariable):
    _nonvar_fields: Incomplete
    grad_to_source: Incomplete
    tensor_to_source: Incomplete
    static_tensor_names: Incomplete
    def __init__(self, value, grad_to_source=None, static_tensor_names=None, tensor_to_source=None, **kwargs) -> None: ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker:
        """This is an optimization to avoid tracing the very slow initialization of the optimizer"""
    def var_getattr(self, tx: InstructionTranslator, name): ...
    def graph_break_if_pending_mutation(self, tx) -> None: ...
    def _set_capturable(self, tx): ...
    def get_python_args(self, *args, **kwargs):
        """Get python values equivalent to the variable tracker args"""
    def move_step_if_cpu(self) -> None: ...
    def map_sources_and_install_guards(self, tx) -> None: ...
    def wrap_tensor(self, tx: InstructionTranslator, tensor_value):
        """Wrap state tensor in a TensorVariable"""
    def update_list_args(self, tx: InstructionTranslator, args, kwargs, py_args, py_kwargs):
        """Update the args and kwargs to the traced optimizer call"""
    def create_finalizer(self, tx) -> None: ...
