import typing
from . import config as config, decorators as decorators, exc as exc, graph_break_hints as graph_break_hints, trace_rules as trace_rules
from .backends.registry import CompilerFn as CompilerFn
from .bytecode_analysis import remove_dead_code as remove_dead_code, remove_pointless_jumps as remove_pointless_jumps
from .bytecode_transformation import Instruction as Instruction, check_inst_exn_tab_entries_valid as check_inst_exn_tab_entries_valid, is_generator as is_generator, propagate_inst_exn_table_entries as propagate_inst_exn_table_entries, transform_code_object as transform_code_object
from .cache_size import CacheSizeRelevantForFrame as CacheSizeRelevantForFrame, compute_cache_size as compute_cache_size, exceeds_recompile_limit as exceeds_recompile_limit, is_recompilation as is_recompilation
from .eval_frame import TorchPatcher as TorchPatcher, always_optimize_code_objects as always_optimize_code_objects, dynamo_tls as dynamo_tls, skip_code as skip_code
from .exc import BackendCompilerFailed as BackendCompilerFailed, FailOnRecompileLimitHit as FailOnRecompileLimitHit, InternalTorchDynamoError as InternalTorchDynamoError, PackageError as PackageError, RecompileLimitExceeded as RecompileLimitExceeded, ShortenTraceback as ShortenTraceback, SkipCodeRecursiveException as SkipCodeRecursiveException, TorchRuntimeError as TorchRuntimeError, UncapturedHigherOrderOpError as UncapturedHigherOrderOpError, Unsupported as Unsupported, augment_exc_message as augment_exc_message, format_error_msg as format_error_msg, unimplemented_v2 as unimplemented_v2
from .guards import CheckFunctionManager as CheckFunctionManager, GuardedCode as GuardedCode, get_and_maybe_log_recompilation_reasons as get_and_maybe_log_recompilation_reasons
from .hooks import Hooks as Hooks
from .output_graph import OutputGraph as OutputGraph
from .package import CompilePackage as CompilePackage
from .pgo import log_frame_dynamic_whitelist as log_frame_dynamic_whitelist, put_code_state as put_code_state
from .replay_record import ExecutionRecord as ExecutionRecord
from .repro.after_dynamo import WrapBackendDebug as WrapBackendDebug
from .resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX as TORCH_DYNAMO_RESUME_IN_PREFIX
from .symbolic_convert import DistributedState as DistributedState, ExceptionStack as ExceptionStack, InstructionTranslator as InstructionTranslator, LocalState as LocalState, SpeculationLog as SpeculationLog
from .trace_rules import is_numpy as is_numpy
from .types import BytecodeHook as BytecodeHook, CacheEntry as CacheEntry, ConvertFrameReturn as ConvertFrameReturn, DynamoFrameType as DynamoFrameType, FrameAction as FrameAction, FrameExecStrategy as FrameExecStrategy, wrap_guarded_code as wrap_guarded_code
from .utils import CleanupManager as CleanupManager, CompileTimeInstructionCounter as CompileTimeInstructionCounter, LazyString as LazyString, chromium_event_timed as chromium_event_timed, counters as counters, dynamo_timed as dynamo_timed, format_bytecode as format_bytecode, gen_record_file_name as gen_record_file_name, get_metrics_context as get_metrics_context, increment_frame as increment_frame, is_namedtuple as is_namedtuple, istype as istype, maybe_disable_inference_mode as maybe_disable_inference_mode, maybe_disable_inference_mode_for_fake_prop as maybe_disable_inference_mode_for_fake_prop, orig_code_map as orig_code_map, reset_graph_break_dup_checker as reset_graph_break_dup_checker, setup_compile_debug as setup_compile_debug, to_int_us as to_int_us, troubleshooting_url as troubleshooting_url, write_record_to_file as write_record_to_file
from .variables.builder import FrameStateSizeEntry as FrameStateSizeEntry
from .variables.torch_function import torch_function_mode_stack_state_mgr as torch_function_mode_stack_state_mgr
from _typeshed import Incomplete
from torch._C._dynamo.guards import GlobalStateGuard as GlobalStateGuard
from torch._dynamo.callback import CallbackTrigger as CallbackTrigger
from torch._dynamo.distributed import get_compile_pg as get_compile_pg
from torch._dynamo.symbolic_convert import TensorifyState as TensorifyState
from torch._guards import CompileContext as CompileContext, CompileId as CompileId, compile_context as compile_context, tracing as tracing
from torch._logging import structured as structured
from torch._utils_internal import compile_time_strobelight_meta as compile_time_strobelight_meta, justknobs_check as justknobs_check, maybe_upload_prof_stats_to_manifold as maybe_upload_prof_stats_to_manifold, signpost_event as signpost_event
from torch.fx._lazy_graph_module import _use_lazy_graph_module as _use_lazy_graph_module
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError as ConstraintViolationError, GuardOnDataDependentSymNode as GuardOnDataDependentSymNode
from torch.monitor import _WaitCounter as _WaitCounter
from torch.nn.parallel.distributed import DistributedDataParallel as DistributedDataParallel
from torch.utils._python_dispatch import _disable_current_modes as _disable_current_modes, is_in_torch_dispatch_mode as is_in_torch_dispatch_mode
from torch.utils._traceback import CapturedTraceback as CapturedTraceback, format_traceback_short as format_traceback_short
from torch.utils.hooks import RemovableHandle as RemovableHandle
from types import CellType, CodeType, FunctionType, ModuleType
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec
from weakref import ReferenceType

np: ModuleType | None
log: Incomplete
bytecode_log: Incomplete
graph_break_log: Incomplete
compile_lock: Incomplete
_T = TypeVar('_T')
_P = ParamSpec('_P')

class TODO_UNKNOWN: ...

class Tracker:
    seen: list[ReferenceType[CodeType]]
    seen_ids: set[int]
    def __init__(self) -> None: ...
    def add(self, strong_obj: CodeType) -> None: ...
    def __contains__(self, item: CodeType) -> bool: ...
    def clear(self) -> None: ...

input_codes: Incomplete
output_codes: Incomplete
initial_global_state: GlobalStateGuard | None

def fx_forward_from_src_skip_result(src: str, globals: dict[str, Any], co_fields: dict[str, str] | None = None) -> FunctionType: ...
def preserve_global_state(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """
    Context manager to:
        1) Save/restore torch.is_grad_enabled() state
        2) Save/restore python random state
        3) Save/restore torch random state
        4) Monkey patch torch.fx.graph_module._forward_from_src
    """
@TorchPatcher.suppress_torch_distributed_warnings
def has_tensor_in_frame(frame: DynamoFrameType) -> bool:
    """Check if the frame has torch.* related bits"""
def exception_handler(e: Exception, code: CodeType, frame: DynamoFrameType | None = None, export: bool = False) -> None: ...

FRAME_COUNTER: int
FRAME_COMPILE_COUNTER: typing.Counter[int | FrameStateSizeEntry]

def maybe_cprofile(func: Callable[_P, _T]) -> Callable[_P, _T]: ...
def cprofile_wrapper(func: Callable[_P, _T]) -> Callable[_P, _T]: ...

class ConvertFrameAssert:
    _torchdynamo_orig_callable: Incomplete
    _one_graph: Incomplete
    _export: Incomplete
    _export_constraints: Incomplete
    _package: Incomplete
    def __init__(self, compiler_fn: CompilerFn, one_graph: bool = True, export: bool = False, export_constraints: typing.Never | None = None, package: CompilePackage | None = None) -> None: ...
    @property
    def _clone_with_backend(self) -> Callable[[CompilerFn], ConvertFrameAssert]: ...
    def __call__(self, frame: DynamoFrameType, cache_entry: CacheEntry | None, hooks: Hooks, frame_state: dict[str, int | FrameStateSizeEntry], *, skip: int = 0) -> ConvertFrameReturn: ...

def convert_frame_assert(compiler_fn: CompilerFn, one_graph: bool = True, export: bool = False, export_constraints: typing.Never | None = None, package: CompilePackage | None = None) -> ConvertFrameAssert:
    """Fully convert a frame into an FX graph"""

_bytecode_hooks: dict[int, BytecodeHook]

def register_bytecode_hook(hook: BytecodeHook) -> RemovableHandle:
    """Register hooks for bytecode generated by Dynamo. The hook can do some
    logging, as well as return a new code object to be used. Please refer
    to `BytecodeHook` for the hook signature.
    """
def _compile(code: CodeType, globals: dict[str, object], locals: dict[str, object], builtins: dict[str, object], closure: tuple[CellType], compiler_fn: CompilerFn, one_graph: bool, export: bool, export_constraints: typing.Never | None, hooks: Hooks, cache_entry: CacheEntry | None, cache_size: CacheSizeRelevantForFrame, frame: DynamoFrameType | None = None, frame_state: dict[str, int | FrameStateSizeEntry] | None = None, *, compile_id: CompileId, skip: int = 0, package: CompilePackage | None = None) -> ConvertFrameReturn: ...

class ConvertFrame:
    _torchdynamo_orig_callable: Incomplete
    _inner_convert: Incomplete
    _hooks: Incomplete
    def __init__(self, compiler_fn: CompilerFn, hooks: Hooks, package: CompilePackage | None = None) -> None: ...
    @property
    def _clone_with_backend(self) -> Callable[[WrapBackendDebug], ConvertFrame]: ...
    def __call__(self, frame: DynamoFrameType, cache_entry: CacheEntry | None, hooks: Hooks, frame_state: dict[str, int | FrameStateSizeEntry], skip: int = 0) -> ConvertFrameReturn: ...

def convert_frame(compiler_fn: CompilerFn, hooks: Hooks, package: CompilePackage | None = None) -> ConvertFrame:
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
def replay(filename: str) -> None: ...
def first_real_inst_idx(code: CodeType) -> int: ...

class ConvertFrameProtocol(typing.Protocol):
    def __call__(self, frame: DynamoFrameType, cache_entry: CacheEntry | None, hooks: Hooks, frame_state: dict[str, int | FrameStateSizeEntry], *, skip: int = 0) -> ConvertFrameReturn: ...

class CatchErrorsWrapper:
    _torchdynamo_orig_callable: Incomplete
    hooks: Incomplete
    def __init__(self, callback: ConvertFrameProtocol, hooks: Hooks) -> None: ...
    def __call__(self, frame: DynamoFrameType, cache_entry: CacheEntry | None, frame_state: dict[str, int | FrameStateSizeEntry]) -> ConvertFrameReturn: ...

def catch_errors_wrapper(callback: ConvertFrameProtocol, hooks: Hooks) -> CatchErrorsWrapper: ...
