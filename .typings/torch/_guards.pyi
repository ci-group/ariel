import abc
import contextlib
import dataclasses
import enum
import sympy
import torch
import traceback
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from torch.utils._backport_slots import dataclass_slots as dataclass_slots
from torch.utils._traceback import CapturedTraceback as CapturedTraceback, format_frame as format_frame
from torch.utils.weak import WeakTensorKeyDictionary as WeakTensorKeyDictionary
from types import CodeType
from typing import Any, Callable, Generic, NamedTuple, TypeVar

log: Incomplete
COMPILE_ID_PATTERN: Incomplete
CA_COMPILE_ID_PATTERN: Incomplete

@dataclass(frozen=True)
class CompileId:
    frame_id: int | None
    frame_compile_id: int | None
    compiled_autograd_id: int | None = ...
    def __str__(self) -> str: ...
    @classmethod
    def from_string(cls, compile_id: str | None):
        """
        Factory method that creates a CompileId from its string representation.
        Keep this in sync with the __str__ method.
        """

class TraceId(NamedTuple):
    compile_id: CompileId
    attempt: int
    def __str__(self) -> str: ...

class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_SPECIALIZED_NN_MODULE = 2
    GLOBAL_SPECIALIZED_NN_MODULE = 3
    CONSTANT = 4
    RANDOM_VALUE = 5
    SHAPE_ENV = 6
    LOCAL_FSDP_MODULE = 7
    GLOBAL_FSDP_MODULE = 8
    BACKWARD_STATE = 9
    EPHEMERAL = 10
    SYNTHETIC_LOCAL = 11
    LOCAL_UNSPECIALIZED_NN_MODULE = 12
    GLOBAL_UNSPECIALIZED_NN_MODULE = 13
    LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE = 14
    GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE = 15
    def is_fsdp_module(self) -> bool: ...
    def is_specialized_nn_module(self) -> bool: ...
    def is_unspecialized_nn_module(self) -> bool: ...
    def is_unspecialized_builtin_nn_module(self) -> bool: ...
    def is_local(self): ...

class GuardBuilderBase: ...

@dataclasses.dataclass(frozen=True)
class SLoc:
    framework_loc: traceback.FrameSummary | str | None
    maybe_user_loc: str | None
    def __str__(self) -> str: ...

class ShapeGuard(NamedTuple):
    expr: sympy.logic.boolalg.Boolean
    sloc: SLoc
    size_oblivious: bool

@dataclasses.dataclass
class Guard:
    originating_source: Source
    create_fn: Callable[[GuardBuilderBase, Guard], None]
    guard_types: list[str] | None = ...
    code_list: list[str] | None = ...
    obj_weakref: object | None = ...
    guarded_class_weakref: type | None = ...
    stack: CapturedTraceback | None = ...
    user_stack: traceback.StackSummary | None = ...
    _hash: int | None = ...
    def __hash__(self): ...
    def sort_key(self): ...
    def __lt__(self, other): ...
    def inner_create_fn(self): ...
    @property
    def name(self) -> str: ...
    @property
    def source(self) -> GuardSource: ...
    @staticmethod
    def weakref_to_str(obj_weakref):
        """
        This is a workaround of a Python weakref bug.

        `obj_weakref` is instance returned by `weakref.ref`,
        `str(obj_weakref)` is buggy if the original obj overrides __getattr__, e.g:

            class MyConfig(dict):
                def __getattr__(self, x):
                    return self[x]

            obj = MyConfig(offset=5)
            obj_weakref = weakref.ref(obj)
            str(obj_weakref)  # raise error: KeyError: '__name__'
        """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def create(self, builder: GuardBuilderBase): ...
    def is_specialized_nn_module(self): ...
    def is_fsdp_module(self): ...
    def is_local(self): ...
    def set_export_info(self, guard_type, guarded_class, code_list, obj_weakref) -> None: ...
T = TypeVar('T')

@dataclasses.dataclass(frozen=True)
class GuardEnvExpr: ...

@dataclasses.dataclass(frozen=True)
class DuplicateInputs(GuardEnvExpr):
    input_source_a: Source
    input_source_b: Source
    def __post_init__(self) -> None: ...

@dataclasses.dataclass(frozen=True)
class StorageOverlap(GuardEnvExpr):
    overlapping_sources: list[Source]
    non_overlapping_sources: list[Source]

class Checkpointable(Generic[T], metaclass=abc.ABCMeta):
    @abstractmethod
    def copy_graphstate(self) -> T: ...
    @abstractmethod
    def restore_graphstate(self, state: T): ...

class GuardsCheckpointState:
    """
    The GuardCheckpointState - it is the T of Checkpointable[T] for GuardsContext
    """
    dynamo_guards: set[Guard]
    def __init__(self, dynamo_guards) -> None: ...
    def diff(self, other):
        """
        Produces a delta against another GuardsCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        Guard type objects.
        """
    def __eq__(self, other): ...

class ModuleContextCheckpointState:
    nn_modules: dict[str, torch.nn.Module]
    def __init__(self, nn_modules) -> None: ...
    def diff(self, other):
        """
        Produces a delta against another ModuleContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        module key names.
        """
    def __eq__(self, other): ...

class ModuleContext(Checkpointable[ModuleContextCheckpointState]):
    nn_modules: dict[str, Any]
    def __init__(self) -> None: ...
    def copy_graphstate(self): ...
    def restore_graphstate(self, state) -> None: ...

class GlobalContextCheckpointState:
    global_state: dict[str, tuple[Callable, ...]]
    def __init__(self, global_states) -> None: ...
    def diff(self, other):
        """
        Produces a delta against another GlobalContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        global key names.
        """
    def __eq__(self, other): ...

class GlobalContext(Checkpointable[GlobalContextCheckpointState]):
    """
    This keeps track of the global torch state during tracing of a function.
    For example, torch.is_grad_enabled.
    """
    _supported_global_states: Incomplete
    global_state: dict[str, tuple[Callable, ...]]
    def __init__(self) -> None: ...
    def copy_graphstate(self): ...
    def restore_graphstate(self, state) -> None: ...

class GuardsSet:
    inner: Incomplete
    def __init__(self, inner=None) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __sub__(self, other): ...
    def __bool__(self) -> bool: ...
    def add(self, guard: Guard, *, collect_debug_stack: bool = True, skip: int = 0): ...
    def update(self, *others: set[Guard]): ...
    def remove_guards_with_source(self, source) -> None:
        """Delete all guards that contains a given source"""

class GuardsContext(Checkpointable[GuardsCheckpointState]):
    dynamo_guards: GuardsSet
    aotautograd_guards: list[GuardEnvExpr]
    def __init__(self) -> None: ...
    def copy_graphstate(self): ...
    def restore_graphstate(self, state) -> None: ...

class HopSubgraphCache(metaclass=abc.ABCMeta):
    @abstractmethod
    def add_dynamo_installed_submodule(self, fn_id: int, identifier: str): ...
    @abstractmethod
    def get_dynamo_installed_submodules(self, fn_id: int) -> list[str]: ...
    @abstractmethod
    def add_autograd_key_entry(self, identifier: str, key: Callable): ...
    @abstractmethod
    def get_autograd_key_entry(self, identifier: str): ...
    @abstractmethod
    def add_proxy_dispatch_entry(self, identifier: str, key: Callable): ...
    @abstractmethod
    def get_proxy_dispatch_entry(self, identifier: str): ...
    @abstractmethod
    def add_lazy_bwd_entry(self, identifier: str, tangent_metadata: tuple[object], gmod: torch.fx.GraphModule): ...
    @abstractmethod
    def get_lazy_bwd_entry(self, identifier: str, tangent_metadata: tuple[object]) -> int: ...

class InvokeSubgraphCache(HopSubgraphCache):
    autograd_cache: dict[str, Callable]
    proxy_dispatch_cache: dict[str, Callable]
    dynamo_installed_submodules: dict[int, list[str]]
    lazy_bwd_cache: dict[str, dict[tuple[object], tuple[torch.fx.GraphModule, int]]]
    def __init__(self) -> None: ...
    def add_dynamo_installed_submodule(self, fn_id: int, identifier: str): ...
    def get_dynamo_installed_submodules(self, fn_id: int) -> list[str]: ...
    def add_autograd_key_entry(self, identifier: str, key: Callable): ...
    def get_autograd_key_entry(self, identifier: str): ...
    def add_proxy_dispatch_entry(self, identifier: str, key: Callable): ...
    def get_proxy_dispatch_entry(self, identifier: str): ...
    def add_lazy_bwd_entry(self, identifier: str, tangent_metadata: tuple[object], gmod: torch.fx.GraphModule): ...
    def get_lazy_bwd_entry(self, identifier: str, tangent_metadata: tuple[object]): ...

class HopDispatchSetCache:
    hop_cache_map: Incomplete
    def __init__(self) -> None: ...
    def get_cache(self, op: torch._ops.HigherOrderOperator) -> HopSubgraphCache | None: ...

_TLS: Incomplete

class CompileContext:
    @staticmethod
    def get() -> CompileContext: ...
    @staticmethod
    def try_get() -> CompileContext | None: ...
    compile_id: CompileId | None
    attempt: int
    shape_env_guards: list[str]
    def __init__(self, compile_id) -> None: ...
    @staticmethod
    def current_compile_id(): ...
    @staticmethod
    def current_trace_id(): ...

class TracingContext:
    """
    Provides the currently installed TracingContext, or None.

    Note that it is a staticmethod, and invocations outside of `with tracing()` (see below), are valid but
    will return None.
    """
    @staticmethod
    def try_get() -> TracingContext | None: ...
    @staticmethod
    def get() -> TracingContext: ...
    guards_context: Incomplete
    module_context: Incomplete
    global_context: Incomplete
    previously_inlined_functions: Incomplete
    previously_cleaned_instructions: Incomplete
    fake_mode: Incomplete
    frame_summary_stack: Incomplete
    loc_in_frame: Incomplete
    fw_metadata: Incomplete
    aot_graph_name: Incomplete
    params_flat: Incomplete
    params_flat_unwrap_subclasses: Incomplete
    params_unwrapped_to_flat_index: Incomplete
    output_strides: list[tuple[int, ...] | None] | None
    force_unspec_int_unbacked_size_like: bool
    tensor_to_context: Incomplete
    fakify_first_call: bool
    hop_dispatch_set_cache: Incomplete
    traced_code: list[CodeType]
    def __init__(self, fake_mode) -> None: ...
    def clear(self) -> None: ...
    @staticmethod
    @contextmanager
    def patch(**kwargs) -> Generator[None]: ...
    @staticmethod
    def extract_stack(): ...
    def _populate_loc_in_frame_summary(self): ...
    @staticmethod
    @contextlib.contextmanager
    def clear_frame() -> Generator[None]: ...
    @staticmethod
    @contextlib.contextmanager
    def current_frame(frame_summary) -> Generator[None]: ...
    @staticmethod
    @contextlib.contextmanager
    def report_output_strides() -> Generator[Incomplete]: ...
    @staticmethod
    def set_current_loc(filename, lineno, frame_name) -> None: ...
    @staticmethod
    def get_traced_code(): ...

@contextmanager
def compile_context(context: CompileContext | None): ...
@contextmanager
def tracing(context: TracingContext | None):
    """
    This function installs the passed in tracing context as a dynamic scoped
    global variable.

    Calls to TracingContext.get() while not under a `with tracing()` context
    will return None.
    """

@dataclasses.dataclass(frozen=True)
class Source:
    def is_dict_key(self): ...
    def is_ephemeral(self): ...
    def reconstruct(self, codegen) -> None: ...
    def guard_source(self) -> GuardSource: ...
    def name(self) -> str: ...
    def make_guard(self, fn) -> Guard: ...
    def is_specialized_nn_module(self) -> bool: ...
    def subguards_allowed(self):
        """True if you can guard on attributes of this"""

@dataclasses.dataclass(frozen=True)
class ChainedSource(Source):
    base: Source
    def is_dict_key(self): ...
    def is_ephemeral(self): ...
    def get_base(self) -> Source: ...

def detect_fake_mode(inputs: Any = None):
    '''
    Attempts to "detect" what the current fake mode is.  If there is one ambiently
    available from TracingContext, we preferentially use that.  Otherwise, we
    heuristically detect the fake mode via the following sources, in order of
    priority:

        - Currently active fake mode on stack
        - Fake mode associated with passed in tensors (inputs does not
          have to be flattened)
    '''
def active_fake_mode():
    """
    Inspects the dispatch mode stack for an active fake mode and returns it.
    Returns None if no fake mode is active.
    """
