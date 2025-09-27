import dataclasses
import types
from _typeshed import Incomplete
from torch._C._dynamo.eval_frame import _CacheEntry as CacheEntry, _ExtraState as ExtraState, _FrameExecStrategy as FrameExecStrategy, _PyInterpreterFrame as DynamoFrameType
from torch._guards import CompileId as CompileId, Guard as Guard
from typing import Any, Callable, NamedTuple, Protocol

FrameState = dict[Any, Any]

class GuardFail(NamedTuple):
    reason: str
    orig_code: types.CodeType

@dataclasses.dataclass(frozen=True)
class GuardFilterEntry:
    name: str
    has_value: bool
    value: object
    guard_type: str
    derived_guard_types: tuple[str, ...]
    is_global: bool
    orig_guard: Guard

class GuardFn(Protocol):
    closure_vars: dict[str, object]
    args: list[str]
    code_parts: list[str]
    verbose_code_parts: list[str]
    global_scope: dict[str, object]
    guard_fail_fn: Callable[[GuardFail], None] | None
    cache_entry: CacheEntry | None
    extra_state: ExtraState | None
    def __call__(self, f_locals: dict[str, object]) -> bool: ...

@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    guard_manager: GuardFn
    compile_id: CompileId
    trace_annotation: str = ...

@dataclasses.dataclass
class ConvertFrameReturn:
    frame_exec_strategy: FrameExecStrategy = dataclasses.field(default_factory=Incomplete)
    apply_to_code: bool = ...
    guarded_code: GuardedCode | None = ...

def wrap_guarded_code(guarded_code: GuardedCode) -> ConvertFrameReturn: ...

class DynamoCallbackFn(Protocol):
    def __call__(self, frame: DynamoFrameType, cache_entry: CacheEntry | None, frame_state: FrameState) -> ConvertFrameReturn: ...
DynamoCallback = DynamoCallbackFn | None | bool

class DynamoGuardHook(Protocol):
    def __call__(self, guard_manager: GuardFn, code: types.CodeType, f_locals: dict[str, object], index: int, last: bool) -> None: ...

class DynamoGuardCompleteHook(Protocol):
    def __call__(self, cache_hit: bool) -> bool: ...

class ProfilerStartHook(Protocol):
    def __call__(self, name: str) -> Any: ...

class ProfilerEndHook(Protocol):
    def __call__(self, record: Any) -> None: ...

class BytecodeHook(Protocol):
    def __call__(self, code: types.CodeType, new_code: types.CodeType) -> types.CodeType | None: ...
