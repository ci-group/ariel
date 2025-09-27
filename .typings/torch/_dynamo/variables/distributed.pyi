import torch
from .. import compiled_autograd as compiled_autograd, variables as variables
from .._trace_wrapped_higher_order_op import trace_wrapped as trace_wrapped
from ..exc import unimplemented_v2 as unimplemented_v2
from ..external_utils import call_module_hooks_from_backward_state as call_module_hooks_from_backward_state
from ..guards import GuardBuilder as GuardBuilder, install_guard as install_guard
from ..source import AttrSource as AttrSource
from ..utils import istype as istype
from .base import VariableTracker as VariableTracker
from .constant import ConstantVariable as ConstantVariable, EnumVariable as EnumVariable
from _typeshed import Incomplete
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator
from torch.fx.experimental._backward_state import BackwardState as BackwardState

class DistributedVariable(VariableTracker):
    """
    The base distributed variable that encapsulates common methods
    for the distributed objects (i.e. ProcessGroup, DeviceMesh, etc.).
    Concrete distributed objects could inherit this class and add object
    specific logic.

    i.e. It provides the check on the distributed package existence
    and hold the tracking value for the corresponding distributed object.
    """
    value: Incomplete
    def __init__(self, value, **kwargs) -> None: ...
    def python_type(self): ...
    @staticmethod
    def is_available(): ...

def is_from_local(value): ...
def is_constant_pg_functions(value): ...

class WorldMetaClassVariable(DistributedVariable):
    """
    Tracks torch.distributed.GroupMember and torch.distributed.group, which are
    instances of the metaclass _WorldMeta.
    """
    @classmethod
    def is_group_member_type(cls, value): ...
    def var_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...

class PlacementClassVariable(DistributedVariable):
    @staticmethod
    def is_placement_type(value): ...
    def as_python_constant(self): ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class PlacementVariable(DistributedVariable):
    @staticmethod
    def is_placement(value): ...
    def as_python_constant(self): ...
    def var_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class DeviceMeshVariable(DistributedVariable):
    @staticmethod
    def is_device_mesh(value): ...
    def as_python_constant(self): ...
    def var_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class ProcessGroupVariable(DistributedVariable):
    """
    We don't want a ProcessGroup object to end up in our output graph.

    But it's common for dynamo to intercept a PG that is then used to get info like
    rank() or world_size(), as well as passed to utility functions in distributed_c10d
    which desugar it into plain types like a ranklist and tag.

    For convenience and proper guarding, we construct a variable type.

    TODO: make it possible to use ProcessGroupVariable as input to simple functions
          like _expand_group without dynamo complaining about making a proxy for it.
          It is not a tensor-like type, and we don't want a proxy- but dynamo assumes
          torch library functions are dealing with tensor-like types and would have proxies
          for their args.
    TODO: should we make this inherit VT instead of UDOV? Do we want any of the default behaviors
          or just graph-break whenever one of our special cases is not hit?
    """
    def as_python_constant(self): ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def var_getattr(self, tx: InstructionTranslator, name): ...
    @staticmethod
    def is_process_group(value): ...

class BackwardHookVariable(VariableTracker):
    """
    Handles torch.utils.hooks.BackwardHook for module-level backward
    hooks.
    """
    @staticmethod
    def create(tx, module: VariableTracker, user_hooks: VariableTracker, user_pre_hooks: VariableTracker): ...
    proxy: Incomplete
    module: Incomplete
    user_hooks: Incomplete
    user_pre_hooks: Incomplete
    def __init__(self, proxy: torch.fx.Proxy, module: VariableTracker, user_hooks: VariableTracker, user_pre_hooks: VariableTracker, **options) -> None: ...
    def as_proxy(self): ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def _setup_hook(self, tx: InstructionTranslator, hook_method_name, args): ...
