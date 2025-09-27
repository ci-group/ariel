import atexit
import collections
import contextlib
import dataclasses
import dis
import enum
import functools
import torch
import torch.fx
import types
import weakref
from . import config as config
from .graph_utils import _get_flat_args as _get_flat_args
from _typeshed import Incomplete
from collections import Counter
from collections.abc import Generator, ItemsView, Iterable, Iterator, KeysView, ValuesView
from contextlib import AbstractContextManager, contextmanager
from torch import fx as fx
from torch._C import _instruction_counter as _instruction_counter, _len_torch_function_stack as _len_torch_function_stack, _pop_torch_function_stack as _pop_torch_function_stack, _push_on_torch_function_stack as _push_on_torch_function_stack
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._dynamo.metrics_context import MetricsContext as MetricsContext, RuntimeMetricsContext as RuntimeMetricsContext
from torch._guards import CompileId as CompileId, Source as Source, TracingContext as TracingContext, detect_fake_mode as detect_fake_mode
from torch._logging import LazyString as LazyString
from torch._subclasses import UnsupportedFakeTensorException as UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor, is_fake as is_fake, maybe_get_fake_mode as maybe_get_fake_mode
from torch._subclasses.meta_utils import is_sparse_compressed as is_sparse_compressed
from torch._utils_internal import justknobs_check as justknobs_check, log_chromium_event_internal as log_chromium_event_internal, log_compilation_event as log_compilation_event, record_chromium_event_internal as record_chromium_event_internal, signpost_event as signpost_event
from torch.fx._utils import _format_graph_code as _format_graph_code, lazy_format_graph_code as lazy_format_graph_code
from torch.monitor import _WaitCounter as _WaitCounter
from torch.nn.modules.lazy import LazyModuleMixin as LazyModuleMixin
from torch.utils._triton import has_triton as has_triton, has_triton_package as has_triton_package
from torch.utils.hooks import RemovableHandle as RemovableHandle
from types import CodeType, MethodWrapperType
from typing import Any, Callable, ClassVar, TypeVar, overload
from typing_extensions import Literal, TypeAlias, TypeGuard, TypeIs

NP_SUPPORTED_MODULES: tuple[types.ModuleType, ...]
NP_TO_TNP_MODULE: Incomplete
T = TypeVar('T')
unpatched_nn_module_getattr: Incomplete
unpatched_nn_module_call: Incomplete
unpatched_nn_module_call_impl: Incomplete
counters: collections.defaultdict[str, Counter[str]]
optimus_scuba_log: dict[str, Any]
troubleshooting_url: str
nnmodule_doc_url: str
nnmodule_doc_url_msg: Incomplete
log: Incomplete
compilation_time_metrics: dict[str, list[float]]
cumulative_time_spent_ns: dict[str, float]
timer_counter: Incomplete

class ReInplaceTrigger(enum.Enum):
    AUTO_FUNC_V1 = 1
    AUTO_FUNC_V2 = 2
    TRITON_OPS = 3

class ReinplaceCounters:
    _values: collections.defaultdict[str, int]
    @classmethod
    def add_missed_bytes(cls, trigger: ReInplaceTrigger, bytes: int): ...
    @classmethod
    def add_missed_opportunities(cls, trigger: ReInplaceTrigger, count: int): ...
    @classmethod
    def clear(cls) -> None: ...
    @classmethod
    def get_total_missed(cls): ...
    @classmethod
    def get_total_missed_bytes(cls): ...
    @classmethod
    def log(cls) -> None: ...

def tabulate(rows: list[tuple[str, object]] | list[list[object]], headers: tuple[str, ...] | list[str]) -> str: ...

curr_frame: int

def increment_frame() -> None: ...
def reset_frame_count() -> None: ...

op_count: int

def increment_op_count(cnt: int) -> None: ...
def calculate_time_spent() -> dict[str, float]: ...
def print_time_report() -> None: ...

_METRICS_CONTEXT: MetricsContext
_RUNTIME_METRICS_CONTEXT: RuntimeMetricsContext

def get_metrics_context() -> MetricsContext: ...
def get_runtime_metrics_context() -> RuntimeMetricsContext: ...

class CompileEventLogLevel(enum.Enum):
    '''
    Enum that loosely corresponds with a "log level" of a given event.

    CHROMIUM_EVENT: Logs only to tlparse.
    COMPILE_EVENT: Logs to tlparse + PT2 Compile Events
    COMPILATION_METRIC: Logs to tlparse, PT2 Compile Events, and dynamo_compile
    '''
    CHROMIUM = 1
    PT2_COMPILE = 2
    COMPILATION_METRIC = 3

class CompileEventLogger:
    '''
    Helper class for representing adding metadata(i.e. columns) to various compile events.
    Use CompileEventLogger to add event data to:
    - Chromium events
    - PT2 Compile Events
    - CompilationMetrics

    This should be used in conjunction with dynamo_timed() and metrics contexts, which create
    timed spans and events. CompileEventLogger uses three log levels (described in CompileEventLogLevel),
    where each log level logs to all sources below it in the hierarchy.

    Example usages:
    - I want to log to an existing chromium event within dynamo timed:
    with dynamo_timed("my_event"):
        CompileEventLogger.chromium("my_event", foo=bar)

    - I want to log my event to both chromium + pt2_compile_events:
    with dynamo_timed("my_event", log_pt2_compile_event=True):
        CompileEventLogger.pt2_compile("my_event", foo=bar)

    - I want to add information to dynamo events and dynamo_compile
        CompileEventLogger.compilation_metric(foo=bar)
    '''
    @staticmethod
    def log_instant_event(event_name: str, metadata: dict[str, Any], time_ns: int | None = None, log_level: CompileEventLogLevel = ...): ...
    @staticmethod
    def add_data(event_name: str, log_level: CompileEventLogLevel, overwrite: bool = False, **metadata: object):
        '''
        Centralized API for adding data to various events
        Log an event to a toplevel "dynamo" event or metrics context
        depending on log level.
        '''
    @staticmethod
    def add_toplevel(log_level: CompileEventLogLevel, overwrite: bool = False, **metadata: object):
        """
        Syntactic sugar for logging to the toplevel event
        """
    @staticmethod
    def increment(event_name: str, log_level: CompileEventLogLevel, key: str, value: int):
        """
        Increments an existing field, or adds it
        """
    @staticmethod
    def increment_toplevel(key: str, value: int = 1, log_level: CompileEventLogLevel = ...):
        """
        Increments a value on the toplevel metric. By default, logs to metric.
        """
    @staticmethod
    def add_to_set(event_name: str, log_level: CompileEventLogLevel, key: str, value: Any):
        """
        Add metadata <value> to a set of values with key <key>. Creates a set if it doesn't exist.
        """
    @staticmethod
    def add_to_set_toplevel(key: str, value: Any, log_level: CompileEventLogLevel = ...):
        """
        Same as add to set, just does it automatically to the toplevel event instead of having to explicitly name it.
        Defaults to COMPILATION_METRIC log level.
        """
    @staticmethod
    def chromium(event_name: str, **metadata: object):
        """
        Add <metadata> to <event_name> in chromium. Each key/value of metadata will appear in the chromium trace.
        <event_name> should be the name of a timed event span passed to `dynamo_timed`.
        """
    @staticmethod
    def pt2_compile(event_name: str, **metadata: object):
        """
        Add <metadata> to <event_name> in chromium and PT2 Compile Events.
        Each key/value of metadata will appear in the chromium trace. Each kwarg name becomes
        a column in PT2 Compile Events, with the corresponding kwarg value.
        <event_name> should be the name of a timed event span passed to `dynamo_timed`,
        with log_to_pt2_compile_events=True.
        """
    @staticmethod
    def compilation_metric(overwrite: bool = False, **metadata: object):
        """
        Add <metadata> to the CompilationMetrics context. Also logs to PT2 Compile Events
        and chromium.
        Each key/value of metadata will appear in the chromium trace. Each kwarg name becomes
        a column in PT2 Compile Events and Dynamo Compile, with the corresponding kwarg value.
        """
    @staticmethod
    def instant(event_name: str, metadata: dict[str, Any], time_ns: int | None = None):
        """
        Log an instant event to chromium logs with name <event_name> at time <time_ns>. The `args` field in
        Perfetto will point to metadata. <time_ns> should be a value obtained from time.time_ns().
        """
    @staticmethod
    def try_add_pt2_compile(event_name: str, **metadata: object):
        """
        Adds to an existing pt2_compile event, but silently returns if the event doesn't exist
        or ChromiumEventLogger is not initialized.
        This function is syntactic sugar for chromium_event_logger().try_add_event_data.
        """
    @staticmethod
    def try_(method_fn, *args, **kwargs) -> None:
        """
        Special function that quietly runs a given method, returning if CHROMIUM_EVENT_LOG is None or metrics context is not set
        """

_dynamo_timed_tls: Incomplete

@contextmanager
def dynamo_timed(key: str, phase_name: str | None = None, log_pt2_compile_event: bool = False, metadata: dict[str, object] | None = None, dynamo_compile_column_us: str | None = None, compile_id: CompileId | None = None, is_backward: bool | None = None, log_waitcounter: bool = False, waitcounter_name_override: str | None = None) -> Generator[Any, None, None]:
    '''
    dynamo_timed is a context manager
    By wrapping a function in dynamo_timed, we can get a few things:

    1) Optionally log timings to pt2_compile_events.
    2) Optionally log timings to CompilationMetrics (dynamo_compile).
    3) Optionally log chromium events.
    4) Optionally increment a WaitCounter.
    5) Store a record in compilation_time_metrics
       For example:

        def _foo(...):
            with dynamo_timed("_foo"):
                ...

        Would show up as an entry in our timing dict:
        OrderedDict([(\'_foo\', [0.083690, 0.23949, 3.1425e-05])])
        This is extremely useful for granular debugging.

    Although it is tempting to use dynamo_timed as a decorator, please do not.
    In its decorator form it makes cProfile traces less useful as dynamo_timed
    suddenly becomes a bottleneck for lots of function calls (as only one parent
    pointer is recorded).

    Params:
    - key: key into compile_time_metrics. If phase_name is not provided, this is
      also the event name used for pt2_compile_events logs and chromium events.
    - phase_name: Optional override for the event name.
    - log_pt2_compile_event: Whether to log a pt2 compile event internally.
    - metadata: Extra metadata to put in pt2_compile_events.
    - dynamo_compile_column_us: If provided, updates the specified CompilationMetrics
      field to be logged to dyname_compile column. We expect all columns to be _us;
      therefore, the field name must end with "_us".
    - compile_id: In the typical case, this parameter should not be needed. Use to
      supply the compile_id for those cases where we want to log a compile_id where
      it\'s not naturally available, e.g., for runtime autotuning.
    - is_backward: Specify forward/backward directly when not available in a
      CompileContext, e.g., during runtime autotuning.
      that support it.
    - log_waitcounter: If set, we\'ll log a waitcounter of the form "pytorch.dynamo_timed.{key}"
    '''
@overload
def compile_times(repr: Literal['str'], aggregate: bool = False) -> str: ...
@overload
def compile_times(repr: Literal['csv'], aggregate: bool = False) -> tuple[list[str], list[object]]: ...
@atexit.register
def dump_compile_times() -> None: ...

tensortype_to_dtype: Incomplete

class DuplicateWarningChecker:
    maxsize: Incomplete
    def __init__(self, maxsize: int = 4096) -> None: ...
    set: Incomplete
    def reset(self) -> None: ...
    def add(self, key: str | tuple[object, object]) -> bool: ...

graph_break_dup_warning_checker: Incomplete

def setup_compile_debug(): ...
def reset_graph_break_dup_checker() -> None: ...
def add_file_handler(): ...
def setup_log_file(): ...
def gen_record_file_name(exc, code) -> str: ...
def write_record_to_file(filename: str, exec_record) -> None: ...
def count_calls(g: fx.Graph) -> int: ...
def identity(x: T) -> T: ...
def hashable(x): ...
def nothing(*args, **kwargs) -> None: ...

class ExactWeakKeyDictionary:
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""
    values: Incomplete
    refs: Incomplete
    def __init__(self) -> None: ...
    def __getitem__(self, key): ...
    def get(self, key, default=None): ...
    def __contains__(self, key) -> bool: ...
    def __setitem__(self, key, value) -> None: ...
    def _remove_id(self, idx) -> None: ...
    def clear(self) -> None: ...

@overload
def istype(obj: object, allowed_types: type[T]) -> TypeIs[T]: ...
@overload
def istype(obj: object, allowed_types: tuple[type[list[T]], type[tuple[T, ...]]]) -> TypeIs[T]: ...
@overload
def istype(obj: object, allowed_types: Iterable[type]) -> bool: ...

_builtin_final_typing_classes: Incomplete

def is_typing(value): ...
def is_numpy_int_type(value): ...
def is_numpy_float_type(value): ...
@overload
def is_lru_cache_wrapped_function(value: Callable[..., T]) -> TypeGuard[functools._lru_cache_wrapper[T]]: ...
@overload
def is_lru_cache_wrapped_function(value: Any) -> TypeGuard[functools._lru_cache_wrapper[Any]]: ...
_FuncTypes: TypeAlias = types.FunctionType | types.BuiltinFunctionType | types.MethodDescriptorType | types.WrapperDescriptorType

def is_function_or_wrapper(value: Any) -> TypeIs[_FuncTypes | torch._ops.OpOverloadPacket | torch._ops.OpOverload]: ...
def is_function(value: Any) -> TypeIs[_FuncTypes]: ...

cmp_name_to_op_mapping: Incomplete
cmp_name_to_op_str_mapping: Incomplete

def is_wrapper_or_member_descriptor(value: Any) -> TypeIs[types.GetSetDescriptorType | types.MethodDescriptorType | types.WrapperDescriptorType | types.MemberDescriptorType | types.MethodWrapperType]: ...
def unwrap_if_wrapper(fn): ...
def unwrap_with_attr_name_if_wrapper(fn): ...
def is_numpy_ndarray(value): ...
def istensor(obj):
    """Check of obj is a tensor"""
def is_lazy_module(mod): ...
def print_once(*args) -> None: ...
def make_cell(val=None):
    """Some black magic to create a cell object that usually only exists in a closure"""
def proxy_args_kwargs(args, kwargs): ...
def to_int_ms(v: float | None) -> int | None: ...
def to_int_us(v: float | None) -> int | None: ...

LOG_FORMAT_VERSION: int

@dataclasses.dataclass
class CompilationMetrics:
    compile_id: str | None = ...
    frame_key: str | None = ...
    co_name: str | None = ...
    co_filename: str | None = ...
    co_firstlineno: int | None = ...
    cache_size: int | None = ...
    accumulated_cache_size: int | None = ...
    guard_count: int | None = ...
    shape_env_guard_count: int | None = ...
    graph_op_count: int | None = ...
    graph_node_count: int | None = ...
    graph_input_count: int | None = ...
    start_time: float | None = ...
    entire_frame_compile_time_s: float | None = ...
    backend_compile_time_s: float | None = ...
    inductor_compile_time_s: float | None = ...
    code_gen_time_s: float | None = ...
    fail_type: str | None = ...
    fail_reason: str | None = ...
    fail_user_frame_filename: str | None = ...
    fail_user_frame_lineno: int | None = ...
    non_compliant_ops: set[str] | None = ...
    compliant_custom_ops: set[str] | None = ...
    restart_reasons: set[str] | None = ...
    dynamo_time_before_restart_s: float | None = ...
    has_guarded_code: bool | None = ...
    remote_cache_time_saved_s: float | None = ...
    structured_logging_overhead_s: float | None = ...
    config_suppress_errors: bool | None = ...
    config_inline_inbuilt_nn_modules: bool | None = ...
    specialize_float: bool | None = ...
    dynamo_config: str | None = ...
    is_forward: bool | None = ...
    num_triton_bundles: int | None = ...
    remote_fx_graph_cache_get_time_ms: int | None = ...
    remote_fx_graph_cache_put_time_ms: int | None = ...
    start_time_us: int | None = ...
    duration_us: int | None = ...
    dynamo_cumulative_compile_time_us: int | None = ...
    aot_autograd_cumulative_compile_time_us: int | None = ...
    inductor_cumulative_compile_time_us: int | None = ...
    inductor_code_gen_cumulative_compile_time_us: int | None = ...
    triton_compile_time_us: int | None = ...
    runtime_cudagraphify_time_us: int | None = ...
    runtime_triton_autotune_time_us: int | None = ...
    dynamo_compile_time_before_restart_us: int | None = ...
    distributed_ephemeral_timeout_us: int | None = ...
    structured_logging_overhead_us: int | None = ...
    remote_fx_graph_cache_get_time_us: int | None = ...
    remote_fx_graph_cache_put_time_us: int | None = ...
    backward_cumulative_compile_time_us: int | None = ...
    end_time_us: int | None = ...
    pre_grad_pass_time_us: int | None = ...
    post_grad_pass_time_us: int | None = ...
    joint_graph_pass_time_us: int | None = ...
    log_format_version: int = ...
    inductor_config: str | None = ...
    remote_cache_version: int | None = ...
    inductor_fx_remote_cache_hit_count: int | None = ...
    inductor_fx_remote_cache_miss_count: int | None = ...
    inductor_fx_remote_cache_backend_type: str | None = ...
    inductor_fx_remote_cache_hit_keys: str | None = ...
    inductor_fx_remote_cache_miss_keys: str | None = ...
    cuda_version: str | None = ...
    triton_version: str | None = ...
    feature_usage: dict[str, bool] | None = ...
    compile_time_autotune_time_us: int | None = ...
    is_runtime: bool | None = ...
    gc_time_us: int | None = ...
    tensorify_float_attempt: bool | None = ...
    tensorify_float_success: bool | None = ...
    tensorify_float_failure: set[str] | None = ...
    guard_latency_us: float | None = ...
    recompile_reason: str | None = ...
    num_graph_breaks: int | None = ...
    triton_kernel_compile_times_us: str | None = ...
    ir_count: int | None = ...
    cudagraph_skip_reason: str | None = ...
    python_version: str | None = ...
    pgo_put_remote_code_state_time_us: int | None = ...
    pgo_get_remote_code_state_time_us: int | None = ...
    param_numel: int | None = ...
    param_bytes: int | None = ...
    param_count: int | None = ...
    @classmethod
    def create(cls, metrics: dict[str, Any]):
        """
        Factory method to create a CompilationMetrics from a dict of fields.
        Includes the logic to add legacy fields and any pre-processing, e.g.,
        we transform some fields to comma-separated strings for scuba logging.
        """

DEFAULT_COMPILATION_METRICS_LIMIT: int
_compilation_metrics: collections.deque[CompilationMetrics]

def add_compilation_metrics_to_chromium(c: CompilationMetrics) -> None:
    """
    These are the common fields in CompilationMetrics that existed before
    metrics_context, and aren't set by MetricsContext.set(). We add the subset
    of them that make sense in `dynamo`/toplevel events in PT2 Compile Events
    directly.

    If you're tempted to add to this list, consider using CompileEventLogger.compilation_metric()
    instead, which will automatically also add it to tlparse and PT2 Compile Events.
    TODO: Get rid of this function and replace it with CompileEventLogger directly instead.
    """
def _get_dynamo_config_for_logging() -> str | None: ...
def _scrubbed_inductor_config_for_logging() -> str | None:
    """
    Method to parse and scrub uninteresting configs from inductor config
    """
def record_compilation_metrics(start_time_ns: int, end_time_ns: int, metrics: dict[str, Any], exc_type: type[BaseException] | None, exc_value: BaseException | None): ...
def set_compilation_metrics_limit(new_size: int) -> None: ...
def clear_compilation_metrics() -> None: ...
def get_compilation_metrics() -> list[CompilationMetrics]: ...

class ChromiumEventLogger:
    """Logs chromium events to structured logs. tlparse will concatenate these into a perfetto UI link.

    See https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.yr4qxyxotyw for
    a specification of the Chromium Event JSON format.
    """
    def get_stack(self) -> list[str]:
        """
        The main event stack, with every chromium event.
        Logged to tlparse.
        """
    def get_outermost_event(self) -> str | None:
        """
        Get the outermost event name (i.e. the longest running event)
        or None if the stack is empty.
        """
    def get_pt2_compile_substack(self):
        """
        A smaller subset of the main stack that gets used to log
        PT2 Compile Events internally.
        """
    def get_event_data(self) -> dict[str, Any]: ...
    tls: Incomplete
    id_: Incomplete
    def __init__(self) -> None: ...
    def try_add_event_data(self, event_name: str, **kwargs) -> None:
        """
        Same as add_event_data, but will silently not log if the event isn't in the stack.
        """
    def add_event_data(self, event_name: str, **kwargs) -> None:
        """
        Adds additional metadata info to an in-progress event
        This metadata is recorded in the END event
        """
    def increment(self, event_name: str, key: str, value: int):
        """
        Increment an integer event data field by the given amount
        """
    def add_to_set(self, event_name: str, key: str, value: Any):
        """
        Add a value to a set within a event_name's metadata if it exists
        """
    def log_event_start(self, event_name: str, time_ns: int, metadata: dict[str, Any], log_pt2_compile_event: bool = False, compile_id: CompileId | None = None) -> None:
        """
        Logs the start of a single event.
        :param str event_name Name of event to appear in trace
        :param time_ns Timestamp in nanoseconds
        :param metadata: Any extra metadata associated with this event
        :param log_pt2_compile_event: If True, log to pt2_compile_events
        :param compile_id: Explicit compile_id (rather than using the current context)
        """
    def reset(self) -> None: ...
    def log_event_end(self, event_name: str, time_ns: int, metadata: dict[str, Any], start_time_ns: int, log_pt2_compile_event: bool, compile_id: CompileId | None = None) -> None:
        """
        Logs the end of a single event. This function should only be
        called after log_event_start with the same event_name.
        :param event_name: Name of event to appear in trace
        :param time_ns: Timestamp in nanoseconds
        :param metadata: Any extra metadata associated with this event
        :param start_time_ns: The start time timestamp in nanoseconds
        :param log_pt_compile_event: If True, log to pt2_compile_events
        :param compile_id: Explicit compile_id (rather than using the current context)
        """
    def _log_timed_event(self, event_name: str, time_ns: int, phase: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Logs a timed event in chromium format. See log_event_start, log_event_end, etc.
        """
    def log_instant_event(self, event_name: str, time_ns: int, metadata: dict[str, Any] | None = None, log_pt2_compile_event: bool = False) -> None:
        """
        Log an instant event with no associated duration.
        :param str event_name: Name of event to appear in trace
        :param int time_ns Timestamp in nanoseconds
        :param Optional[Dict[str, Any]] metadata: Any extra metadata associated with this event
        :param str cname optional color for the arrow in the trace
        """

CHROMIUM_EVENT_LOG: ChromiumEventLogger | None

def get_chromium_event_logger() -> ChromiumEventLogger: ...
def chromium_event_log_active() -> bool: ...
@contextmanager
def chromium_event_timed(event_name: str, reset_event_log_on_exit: bool = False, log_pt2_compile_event: bool = False) -> Generator[Any, None, None]:
    """
    Context manager that creates a chromium start and end event. Chromium event
    logging is integrated with dynamo_timed, so you probably want to use that
    instead. Use this context manager only if you want to avoid dynamo_timed.
    """

@dataclasses.dataclass
class CleanupHook:
    """Remove a global variable when hook is called"""
    scope: dict[str, Any]
    name: str
    def __call__(self, *args) -> None: ...
    @staticmethod
    def create(scope, name, val): ...

class CleanupManager(ExactWeakKeyDictionary):
    count: int
    instance: ClassVar[CleanupManager]
    def _remove_id(self, idx) -> None: ...

def clone_tensor(x):
    """Clone the tensor and its gradient"""
def clone_input(x, *, dtype=None):
    """copy while preserving strides"""
def clone_inputs(example_inputs): ...
def skip_frame_if_in_functorch_mode(val: torch.Tensor): ...
@contextmanager
def preserve_rng_state() -> Generator[None]: ...
def is_jit_model(model0): ...
def torchscript(model, example_inputs, verbose: bool = False): ...
def getfile(obj): ...
def is_namedtuple(obj):
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
def is_namedtuple_cls(cls):
    """Test if an object is a namedtuple or a (torch.return_types|torch.autograd.forward_ad).* quasi-namedtuple"""
def namedtuple_fields(cls) -> tuple[str, ...]:
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
def checkpoint_params(gm): ...
def timed(model, example_inputs, times: int = 1): ...
def check_is_cuda(gm, example_inputs): ...
def rot_n_helper(n): ...

common_constant_types: set[type]

def is_safe_constant(v): ...
@functools.cache
def common_constants(): ...
def is_torch_sym(value: Any) -> TypeGuard[torch.SymBool | torch.SymInt]: ...
def is_int_specialization_case(value, source): ...
def specialize_symnode(arg): ...
def guard_if_dyn(arg): ...
def check_constant_args(args, kwargs): ...
def check_unspec_python_args(args, kwargs): ...
def check_unspec_or_constant_args(args, kwargs): ...
def check_numpy_ndarray_args(args, kwargs): ...

dict_keys: type[KeysView[Any]]
dict_values: type[ValuesView[Any]]
dict_items: type[ItemsView[Any, Any]]
odict_values: type[ValuesView[Any]]
tuple_iterator: type[Iterator[Any]]
range_iterator: type[Iterator[Any]]
tuple_iterator_len: Incomplete
object_new: Incomplete
dict_new: Incomplete
dict_methods: Incomplete
tuple_new: Incomplete
tuple_methods: Incomplete
list_methods: Incomplete
list_getitem: Incomplete
str_methods: Incomplete

def builtin_dict_keys(d): ...
def get_items_from_dict(obj): ...
def nn_module_new(cls): ...
def product(it): ...
def tuple_iterator_getitem(it, index): ...
def dataclass_fields(cls): ...
iter_next = next

def normalize_range_iter(range_iter) -> tuple[int, int, int]: ...
def to_subclass(t, cls): ...

dict_getitem: Incomplete

def dict_keys_getitem(d, n): ...
def enum_repr(value, local): ...
def set_example_value(node, example_value) -> None: ...
def _get_fake_tensor(vt): ...
def iter_contains(items, search, tx, check_tensor_identity: bool = False): ...
def key_is_id(k: Any) -> TypeIs[torch.Tensor | torch.nn.Module | MethodWrapperType]:
    """Returns whether it indexes dictionaries using its id"""
def key_to_id(value): ...
def const_repr(x, *, local) -> str: ...
def dict_keys_repr(const_keys, *, local) -> str: ...

GLOBAL_KEY_PREFIX: str

def get_safe_global_name(tx, root, obj): ...
def is_in(item: Any, *containers) -> bool: ...
def get_unique_name_wrt(prefix: str, *containers, requires_suffix: bool = False) -> str:
    """
    Return a name that starts with `prefix` and is not in any of the
    `containers` (e.g., map, set).
    """
def wrap_fake_exception(fn): ...
def deepcopy_to_fake_tensor(obj, fake_mode): ...
def rmse(ref, res):
    """
    Calculate root mean squared error
    """
def same(ref, res, fp64_ref=None, cos_similarity: bool = False, tol: float = 0.0001, equal_nan: bool = False, exact_dtype: bool = True, relax_numpy_equality: bool = False, ignore_non_fp: bool = False, log_error=..., use_larger_multiplier_for_smaller_tensor: bool = False, force_max_multiplier: bool = False):
    """Check correctness to see if ref and res match"""
def format_func_info(code): ...
@contextlib.contextmanager
def disable_cache_limit() -> Generator[None]: ...

orig_code_map: Incomplete
guard_failures: collections.defaultdict[Any, list[Any]]
graph_break_reasons: list[torch._dynamo.output_graph.GraphCompileReason]
seen_code_map: Incomplete

@functools.cache
def _get_debug_dir(root_dir): ...
def get_debug_dir(): ...
def extract_fake_example_value(node, required: bool = True): ...
def ensure_graph_fake(e, tx): ...
def get_fake_values_from_nodes(tx, nodes, allow_non_graph_fake): ...
def get_fake_value(node, tx, allow_non_graph_fake: bool = False):
    """
    Run the computation represented by `node` using fake tensors and return the result.

    allow_non_graph_fake: whether to allow the return result to be:
        1. non-fake or 2. fake that is not created by this instance of Dynamo.
        If `True`, you must be prepared to deal with such return values, ideally
        by further wrapping them as this graph's fakes.
    """

_current_node: Incomplete

def get_current_node(): ...
@contextmanager
def set_current_node(node) -> Generator[None]: ...
def run_node(tracer, node, args, kwargs, nnmodule):
    """
    Runs a given node, with the given args and kwargs.

    Behavior is dictated by a node's op.

    run_node is useful for extracting real values out of nodes.
    See get_real_value for more info on common usage.

    Note: The tracer arg is only used for 'get_attr' ops
    Note: The nnmodule arg is only used for 'call_module' ops

    Nodes that are not call_function, call_method, call_module, or get_attr will
    raise an AssertionError.
    """
def get_real_value(node, tracer):
    """
    Run the actual computation represented by `node` and return the result.
    This will execute any dependent nodes in the graph as well.
    """
def assert_no_fake_params_or_buffers(gm): ...
def fqn(obj: Any):
    """
    Returns the fully qualified name of the object.
    """
def ifdynstaticdefault(count1, count2): ...
def import_submodule(mod: types.ModuleType):
    """
    Ensure all the files in a given submodule are imported
    """
def object_has_getattribute(value: Any): ...
def object_setattr_ignore_descriptor(obj, name, value) -> None: ...
def class_has_getattribute(cls): ...
def get_custom_getattr(value: Any, ignore_nn_module_getattr: bool = False): ...

class TensorStaticReason(enum.Enum):
    PARAMETER = 2
    NOT_TENSOR = 4
    NN_MODULE_PROPERTY = 5

def tensor_static_reason_to_message(reason: TensorStaticReason): ...
def tensor_always_has_static_shape(tensor: torch.Tensor | Any, is_tensor: bool, tensor_source: Source) -> tuple[bool, TensorStaticReason | None]:
    '''
    Given a tensor, source, and is_tensor flag, determine if a shape should be static.

    Args:
    tensor - the real tensor to evaluate, parameters force a static shape.
    is_tensor - internal dynamo check, essentially "is_tensor": target_cls is TensorVariable,
    tensors not in a TensorVariable for whatever reason are forced static.

    Returns a tuple, where the first element is the bool of whether or not this tensor should have a static shape.
    The second element is a TensorStaticReason, useful for passing to tensor_static_reason_to_message if needed.
    '''
def lazy_format_graph_tabular(fn_name, gm): ...
def format_bytecode(prefix, name, filename, line_no, code): ...

forward_hook_names: Incomplete
backward_hook_names: Incomplete
state_dict_hook_names: Incomplete
all_hook_names: Incomplete

def nn_module_has_global_hooks(): ...
def nn_module_get_all_hooks(mod, check_forward_hooks: bool = False, check_backward_hooks: bool = False, check_state_dict_hooks: bool = False):
    """
    Sometimes its useful to differentiate between types of hooks such as forward/backward/pre
    hooks executed during module.__call__, and state_dict hooks which are executed separately.
    """
def nnmodule_has_hooks(mod, check_forward_hooks: bool = False, check_backward_hooks: bool = False, check_state_dict_hooks: bool = False):
    """
    Helper function to check if a module has any hooks attached to it.
    """
def to_numpy_helper(value):
    """Convert tensor and tnp.ndarray to numpy.ndarray."""
def numpy_to_tensor(value):
    """Convert tnp.ndarray to tensor, leave other types intact. If a list/tuple, loop through it to convert."""

class numpy_to_tensor_wrapper:
    f: Incomplete
    __name__: Incomplete
    def __init__(self, f) -> None: ...
    def __repr__(self) -> str: ...
    def __call__(self, *args, **kwargs): ...

def numpy_attr_wrapper(obj, name): ...

class numpy_method_wrapper:
    """Convert obj from torch.Tensor to tnp.ndarray and call method. Then convert result back to torch.Tensor."""
    method: Incomplete
    __name__: Incomplete
    def __init__(self, method: str) -> None: ...
    def __repr__(self) -> str: ...
    def __call__(self, *args, **kwargs): ...

class numpy_operator_wrapper:
    """Implements dunder methods for tnp.ndarray via functions from the operator library"""
    op: Incomplete
    __name__: Incomplete
    def __init__(self, op: Callable[..., Any]) -> None: ...
    def __repr__(self) -> str: ...
    def __call__(self, *args, **kwargs): ...

def defake(x): ...
def _disable_side_effect_safety_checks_for_current_subtracer(fn, *args, **kwargs): ...
def is_utils_checkpoint(obj): ...
def is_invoke_subgraph(obj): ...
def build_invoke_subgraph_variable(**options): ...
def build_checkpoint_variable(**options): ...
def is_compile_supported(device_type): ...
def _fix_offset(str: str, offset: int) -> int:
    """
    Convert byte offset `offset` of `str` into character offset.
    Byte offset is used for 3.11+ instruction column data.
    Takes things like unicode characters into consideration.

    Unchanged from CPython implementation.
    """

@dataclasses.dataclass
class _Anchors:
    left_end_lineno: int
    left_end_offset: int
    right_start_lineno: int
    right_start_offset: int

def _extract_anchors_from_expr(segment: str) -> _Anchors | None:
    """
    Given source code `segment` corresponding to a bytecode
    instruction, determine:
        - for binary ops, the location of the binary op
        - for indexing, the location of the brackets.
    `segment` is expected to be a valid Python expression
    """
def get_instruction_source_311(code: types.CodeType, inst: dis.Instruction) -> str:
    """
    Python 3.11+ only. Returns lines of source code (from code object `code`)
    corresponding to `inst`'s location data, and underlines relevant code to `inst`.

    Example: CALL on `g`:
    f(g(
      ^^
        h(x)))
        ^^^^^

    We need our own implementation in < 3.13 since `format_frame_summary` in
    Python's `traceback` module doesn't handle multi-line expressions
    (and their anchor extraction code is not completely correct).
    """
def get_static_address_type(t): ...
def is_rng_state_getter_or_setter(value): ...
def is_tensor_base_attr_getter(value): ...
def is_tensor_getset_descriptor(name): ...
def is_torch_function_object(value): ...
def has_torch_function(vt: torch._dynamo.variables.base.VariableTracker) -> bool: ...
def to_fake_tensor(t, fake_mode): ...
def is_frozen_dataclass(value): ...
def get_first_attr(obj, *attrs):
    """
    Return the first available attribute or throw an exception if none is present.
    """
@contextlib.contextmanager
def maybe_enable_compiled_autograd(should_enable, fullgraph: bool = True, dynamic: bool = True) -> Generator[Incomplete, None, Incomplete]: ...
def invalid_removeable_handle(): ...
def nn_module_proxy(mod): ...

class GmWrapper(torch.nn.Module):
    gm: Incomplete
    unflatten_fn: Incomplete
    def __init__(self, gm, unflatten_fn) -> None: ...
    def forward(self, *args): ...

def flatten_graph_inputs(gm: torch.fx.GraphModule, inputs, compile_gm):
    """
    Mutate inputs so that they are flat and wrap gm such that it
    accepts those inputs.  This is needed for graphs that take
    bumpy inputs.
    """
def get_locals_to_steal(maybe_gm): ...
def set_locals_to_steal(gm, locals_to_steal) -> None: ...

class Lit:
    s: Incomplete
    def __init__(self, s) -> None: ...
    def __repr__(self) -> str: ...

warn_once_cache: set[str]

def warn_once(msg, stacklevel: int = 1) -> None: ...
def strip_color_from_string(text): ...
@contextlib.contextmanager
def _disable_saved_tensors_hooks_during_tracing() -> Generator[None]: ...
def is_parameter_freezing(): ...
def get_torch_function_mode_stack(): ...
def get_torch_function_mode_stack_at(ind): ...
def set_torch_function_mode_stack(stack) -> None: ...
def clear_torch_function_mode_stack() -> None: ...
def _breakpoint_for_c_dynamo(*args) -> None: ...
def verify_guard_fn_signature(value) -> None: ...
def does_not_override_dict_iter_methods(user_cls): ...
@torch._disable_dynamo
def call_size(x, i): ...
@torch._disable_dynamo
def call_stride(x, i): ...
@torch._disable_dynamo
def call_storage_offset(x): ...
def _extract_tensor_dict(t): ...

user_obj_id_to_weakref: dict[int, weakref.ReferenceType[object]]

def get_user_object_from_id(obj_id): ...
def store_user_object_weakref(obj) -> None: ...

class CompileTimeInstructionCounter:
    _counter: int
    _id: int
    _depth: int
    @classmethod
    def start(cls) -> None: ...
    @classmethod
    def end(cls) -> None: ...
    @classmethod
    def clear(cls) -> None: ...
    @classmethod
    def value(cls) -> int: ...
    @classmethod
    @contextmanager
    def record(cls) -> Generator[None]: ...

def set_feature_use(feature: str, usage: bool):
    """
    Records whether we are using a feature
    Generally a feature is a JK.
    """

_ddp_optimization_mode: tuple[str, ...]

def get_optimize_ddp_mode(): ...
@contextmanager
def maybe_disable_inference_mode() -> Generator[None, None, None]:
    """
    Disables torch.inference_mode for the compilation (still on at runtime).
    This simplifies the compile stack where we can assume that inference_mode
    will always be off.

    Since inference_mode is equivalent to no_grad + some optimizations (version
    counts etc), we turn on no_grad here. The other optimizations are not
    relevant to torch.compile.
    """
@contextmanager
def maybe_disable_inference_mode_for_fake_prop() -> Generator[None, None, None]:
    """
    Turns off tracking of inference_mode for fake tensor propagation. With this
    context manager, when a real tensor is converted to fake tensor, the fake
    tensor looses its inference-ness.
    """
def is_node_meta_valid(node: torch.fx.Node | None) -> bool: ...
@torch._disable_dynamo
def record_pregraph_bytecode_enter() -> AbstractContextManager[None]: ...
@torch._disable_dynamo
def record_pregraph_bytecode_exit(cm: AbstractContextManager[None]) -> None: ...
def get_traced_code() -> list[CodeType]: ...
