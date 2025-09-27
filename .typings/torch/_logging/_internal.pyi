import functools
import logging
from _typeshed import Incomplete
from dataclasses import dataclass, field
from torch._guards import CompileId as CompileId
from torch._utils_internal import log_trace_structured_event as log_trace_structured_event
from torch.utils._traceback import CapturedTraceback as CapturedTraceback
from typing import Any, Callable, Generic
from typing_extensions import ParamSpec

_P = ParamSpec('_P')
log: Incomplete
trace_log: Incomplete
DEFAULT_LOG_LEVEL: Incomplete
LOG_ENV_VAR: str
LOG_OUT_ENV_VAR: str
LOG_FORMAT_ENV_VAR: str
LOG_TRACE_ID_FILTER: str
TRACE_ENV_VAR: str
DTRACE_ENV_VAR: str
LOG_TRACE_HANDLER: LazyTraceHandler | None
GET_DTRACE_STRUCTURED: bool

@dataclass
class LogRegistry:
    log_alias_to_log_qnames: dict[str, list[str]] = field(default_factory=dict)
    artifact_log_qnames: set[str] = field(default_factory=set)
    child_log_qnames: set[str] = field(default_factory=set)
    artifact_names: set[str] = field(default_factory=set)
    visible_artifacts: set[str] = field(default_factory=set)
    artifact_descriptions: dict[str, str] = field(default_factory=dict)
    off_by_default_artifact_names: set[str] = field(default_factory=set)
    artifact_log_formatters: dict[str, logging.Formatter] = field(default_factory=dict)
    def is_artifact(self, name): ...
    def is_log(self, alias): ...
    def register_log(self, alias, log_qnames: str | list[str]) -> None: ...
    def register_artifact_name(self, name, description, visible, off_by_default, log_format) -> None: ...
    def register_artifact_log(self, artifact_log_qname) -> None: ...
    def register_child_log(self, log_qname) -> None: ...
    def get_log_qnames(self) -> set[str]: ...
    def get_artifact_log_qnames(self): ...
    def get_child_log_qnames(self): ...
    def is_off_by_default(self, artifact_qname): ...

@dataclass
class LogState:
    log_qname_to_level: dict[str, str] = field(default_factory=dict)
    artifact_names: set[str] = field(default_factory=set)
    def enable_artifact(self, artifact_name) -> None: ...
    def is_artifact_enabled(self, name): ...
    def enable_log(self, log_qnames, log_level) -> None: ...
    def get_log_level_pairs(self):
        """Returns all qualified module names for which the user requested
        explicit logging settings.

        .. warning:

            This function used to return all loggers, regardless of whether
            or not the user specified them or not; it now only returns logs
            which were explicitly mentioned by the user (and torch, which
            always is implicitly requested when we initialize our logging
            subsystem.)
        """
    def clear(self) -> None: ...

log_registry: Incomplete
log_state: Incomplete
DEFAULT_LOGGING: Incomplete

def set_logs(*, all: int | None = None, dynamo: int | None = None, aot: int | None = None, autograd: int | None = None, dynamic: int | None = None, inductor: int | None = None, distributed: int | None = None, c10d: int | None = None, ddp: int | None = None, fsdp: int | None = None, dtensor: int | None = None, onnx: int | None = None, bytecode: bool = False, aot_graphs: bool = False, aot_joint_graph: bool = False, ddp_graphs: bool = False, graph: bool = False, graph_code: bool = False, graph_code_verbose: bool = False, graph_breaks: bool = False, graph_sizes: bool = False, guards: bool = False, recompiles: bool = False, recompiles_verbose: bool = False, trace_source: bool = False, trace_call: bool = False, trace_bytecode: bool = False, output_code: bool = False, kernel_code: bool = False, schedule: bool = False, perf_hints: bool = False, pre_grad_graphs: bool = False, post_grad_graphs: bool = False, ir_pre_fusion: bool = False, ir_post_fusion: bool = False, onnx_diagnostics: bool = False, fusion: bool = False, overlap: bool = False, export: int | None = None, modules: dict[str, int | bool] | None = None, cudagraphs: bool = False, sym_node: bool = False, compiled_autograd: bool = False, compiled_autograd_verbose: bool = False, cudagraph_static_inputs: bool = False, benchmarking: bool = False, autotuning: bool = False, graph_region_expansion: bool = False, inductor_metrics: bool = False, hierarchical_compile: bool = False) -> None:
    '''
    Sets the log level for individual components and toggles individual log
    artifact types.

    .. warning:: This feature is a prototype and may have compatibility
        breaking changes in the future.

    .. note:: The ``TORCH_LOGS`` environment variable has complete precedence
        over this function, so if it was set, this function does nothing.

    A component is a set of related features in PyTorch. All of the log
    messages emitted from a given component have their own log levels. If the
    log level of a particular message has priority greater than or equal to its
    component\'s log level setting, it is emitted. Otherwise, it is suppressed.
    This allows you to, for instance, silence large groups of log messages that
    are not relevant to you and increase verbosity of logs for components that
    are relevant. The expected log level values, ordered from highest to lowest
    priority, are:

        * ``logging.CRITICAL``
        * ``logging.ERROR``
        * ``logging.WARNING``
        * ``logging.INFO``
        * ``logging.DEBUG``
        * ``logging.NOTSET``

    See documentation for the Python ``logging`` module for more information on
    log levels: `<https://docs.python.org/3/library/logging.html#logging-levels>`_

    An artifact is a particular type of log message. Each artifact is assigned
    to a parent component. A component can emit many different kinds of
    artifacts. In general, an artifact is emitted if either its corresponding
    setting in the argument list below is turned on or if its parent component
    is set to a log level less than or equal to the log level of the artifact.

    Keyword args:
        all (:class:`Optional[int]`):
            The default log level for all components. Default: ``logging.WARN``

        dynamo (:class:`Optional[int]`):
            The log level for the TorchDynamo component. Default: ``logging.WARN``

        aot (:class:`Optional[int]`):
            The log level for the AOTAutograd component. Default: ``logging.WARN``

        autograd (:class:`Optional[int]`):
            The log level for autograd. Default: ``logging.WARN``

        inductor (:class:`Optional[int]`):
            The log level for the TorchInductor component. Default: ``logging.WARN``

        dynamic (:class:`Optional[int]`):
            The log level for dynamic shapes. Default: ``logging.WARN``

        distributed (:class:`Optional[int]`):
            Whether to log c10d communication operations and other debug info from PyTorch Distributed components.
            Default: ``logging.WARN``

        c10d (:class:`Optional[int]`):
            Whether to log c10d communication operations related debug info in PyTorch Distributed components.
            Default: ``logging.WARN``

        ddp (:class:`Optional[int]`):
            Whether to log debug info related to ``DistributedDataParallel``(DDP) from PyTorch Distributed components.
            Default: ``logging.WARN``

        fsdp (:class:`Optional[int]`):
            Whether to log debug info related to ``FullyShardedDataParallel``(FSDP) in PyTorch Distributed components.
            Default: ``logging.WARN``

        dtensor (:class:`Optional[int]`):
            Whether to log debug info related to ``DTensor``(DTensor) in PyTorch Distributed components.
            Default: ``logging.WARN``

        onnx (:class:`Optional[int]`):
            The log level for the ONNX exporter component. Default: ``logging.WARN``

        bytecode (:class:`bool`):
            Whether to emit the original and generated bytecode from TorchDynamo.
            Default: ``False``

        aot_graphs (:class:`bool`):
            Whether to emit the graphs generated by AOTAutograd. Default: ``False``

        aot_joint_graph (:class:`bool`):
            Whether to emit the joint forward-backward graph generated by AOTAutograd. Default: ``False``

        ddp_graphs (:class:`bool`):
            Whether to emit graphs generated by DDPOptimizer. Default: ``False``

        graph (:class:`bool`):
            Whether to emit the graph captured by TorchDynamo in tabular format.
            Default: ``False``

        graph_code (:class:`bool`):
            Whether to emit the python source of the graph captured by TorchDynamo.
            Default: ``False``

        graph_code_verbose (:class:`bool`):
            Whether to emit verbose/intermediate FX pass logs for graph code. Default: ``False``

        graph_breaks (:class:`bool`):
            Whether to emit the graph breaks encountered by TorchDynamo.
            Default: ``False``

        graph_sizes (:class:`bool`):
            Whether to emit tensor sizes of the graph captured by TorchDynamo.
            Default: ``False``

        guards (:class:`bool`):
            Whether to emit the guards generated by TorchDynamo for each compiled
            function. Default: ``False``

        recompiles (:class:`bool`):
            Whether to emit a guard failure reason and message every time
            TorchDynamo recompiles a function. Default: ``False``

        recompiles_verbose (:class:`bool`):
            Whether to emit all guard failure reasons when TorchDynamo recompiles
            a function, even those that are not actually run. Default: ``False``

        trace_source (:class:`bool`):
            Whether to emit when TorchDynamo begins tracing a new line. Default: ``False``

        trace_call (:class:`bool`):
            Whether to emit detailed line location when TorchDynamo creates an FX node
            corresponding to function call. Python 3.11+ only. Default: ``False``

        trace_bytecode (:class:`bool`):
            Whether to emit bytecode instructions and traced stack state as TorchDynamo
            traces bytecode. Default: ``False``

        output_code (:class:`bool`):
            Whether to emit the TorchInductor output code on a per-graph basis. Default: ``False``

        kernel_code (:class:`bool`):
            Whether to emit the TorchInductor output code on a per-kernel bases. Default: ``False``

        schedule (:class:`bool`):
            Whether to emit the TorchInductor schedule. Default: ``False``

        perf_hints (:class:`bool`):
            Whether to emit the TorchInductor perf hints. Default: ``False``

        pre_grad_graphs (:class:`bool`):
            Whether to emit the graphs before inductor grad passes. Default: ``False``

        post_grad_graphs (:class:`bool`):
            Whether to emit the graphs generated by after post grad passes. Default: ``False``

        ir_pre_fusion (:class:`bool`):
            Whether to emit the graphs before inductor fusion passes. Default: ``False``

        ir_post_fusion (:class:`bool`):
            Whether to emit the graphs after inductor fusion passes. Default: ``False``

        onnx_diagnostics (:class:`bool`):
            Whether to emit the ONNX exporter diagnostics in logging. Default: ``False``

        fusion (:class:`bool`):
            Whether to emit detailed Inductor fusion decisions. Default: ``False``

        overlap (:class:`bool`):
            Whether to emit detailed Inductor compute/comm overlap decisions. Default: ``False``

        sym_node (:class:`bool`):
            Whether to emit debug info for various SymNode opterations. Default: ``False``

        export (:class:`Optional[int]`):
            The log level for export. Default: ``logging.WARN``

        benchmarking (:class:`bool`):
            Whether to emit detailed Inductor benchmarking information. Default: ``False``

        modules (dict):
            This argument provides an alternate way to specify the above log
            component and artifact settings, in the format of a keyword args
            dictionary given as a single argument. There are two cases
            where this is useful (1) if a new log component or artifact has
            been registered but a keyword argument for it has not been added
            to this function and (2) if the log level for an unregistered module
            needs to be set. This can be done by providing the fully-qualified module
            name as the key, with the log level as the value. Default: ``None``

        cudagraph_static_inputs (:class:`bool`):
            Whether to emit debug info for cudagraph static input detection. Default: ``False``

        autotuning (:class:`bool`):
            Autotuning choice logs, such as kernel source, perf, and tuning parameters. Default: ``False``

        graph_region_expansion (:class:`bool`):
            Whether to emit the detailed steps of the duplicate graph region tracker expansion algorithm. Default: ``False``

        inductor_metrics (:class:`bool`):
            Whether to estimate the runtimes of the nodes in a graph and log them to the metrics table. Default: ``False``

        hierarchical_compile (:class:`bool`):
            Whether to emit debug info for hierarchical compilation. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> import logging

        # The following changes the "dynamo" component to emit DEBUG-level
        # logs, and to emit "graph_code" artifacts.

        >>> torch._logging.set_logs(dynamo=logging.DEBUG, graph_code=True)

        # The following enables the logs for a different module

        >>> torch._logging.set_logs(modules={"unregistered.module.name": logging.DEBUG})
    '''
def get_loggers() -> list[logging.Logger]:
    """
    Returns: a list of all registered loggers
    """
def register_log(setting_name, log_name) -> None:
    """
    Enables a log to be controlled by the env var and user API with the setting_name
    Args:
        setting_name:  the shorthand name used in the env var and user API
        log_name:  the log name that the setting_name is associated with
    """
def register_artifact(setting_name, description, visible: bool = False, off_by_default: bool = False, log_format=None) -> None:
    """
    Enables an artifact to be controlled by the env var and user API with name
    Args:
        setting_name: the shorthand name used in the env var and user API
        description: A description of what this outputs
        visible: Whether it gets suggested to users by default
        off_by_default: whether this artifact should be logged when the ancestor loggers
            are enabled at level DEBUG
    """
def getArtifactLogger(module_qname, artifact_name) -> logging.Logger: ...

INCR_VERBOSITY_CHAR: str
DECR_VERBOSITY_CHAR: str
VERBOSITY_REGEX: Incomplete

def configure_artifact_log(log) -> None: ...
def _gen_settings_regex(): ...
def _validate_settings(settings): ...
def help_message(verbose: bool = False): ...
def _invalid_settings_err_msg(settings, verbose: bool = False): ...
@functools.lru_cache
def _parse_log_settings(settings): ...
def _is_valid_module(qname): ...
def _update_log_state_from_env() -> None: ...
def _has_registered_parent(log_qname) -> bool: ...
def make_module_path_relative(abs_path):
    """
    Given an absolute filepath corresponding to a Python module which was
    loaded via normal import mechanisms using sys.path, convert it into
    a relative path relative to one of the Python search paths.
    """

class TorchLogsFormatter(logging.Formatter):
    _is_trace: Incomplete
    _trace_id_filter: Incomplete
    def __init__(self, *, trace: bool = False, trace_id_filter: set[str] | None = None) -> None: ...
    def format(self, record): ...

def _default_formatter(): ...

DEFAULT_FORMATTER: Incomplete

def _setup_handlers(create_handler_fn, log) -> None: ...

handlers: Incomplete

def _track_handler(handler): ...
def _is_torch_handler(handler): ...
def _clear_handlers(log) -> None: ...
def _reset_logs() -> None: ...
def _get_log_state(): ...
def _set_log_state(state) -> None: ...
def _init_logs(log_file_name=None) -> None: ...

class LazyTraceHandler(logging.StreamHandler):
    """Like FileHandler, but the file is allocated lazily only upon the first log message"""
    root_dir: Incomplete
    stream: Incomplete
    _builtin_open: Incomplete
    def __init__(self, root_dir: str | None) -> None: ...
    def close(self) -> None: ...
    def emit(self, record) -> None: ...

@functools.cache
def warning_once(logger_obj, *args, **kwargs) -> None:
    """
    This function is similar to `logger.warning()`, but will emit the warning with the same message only once
    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """

class LazyString(Generic[_P]):
    func: Incomplete
    args: Incomplete
    kwargs: Incomplete
    def __init__(self, func: Callable[_P, str], *args: _P.args, **kwargs: _P.kwargs) -> None: ...
    def __str__(self) -> str: ...

structured_logging_overhead: dict[str, float]

def add_structured_logging_overhead(time_spent: float) -> None: ...
def get_structured_logging_overhead() -> float | None: ...
def trace_structured_artifact(name: str, encoding: str, payload_fn: Callable[[], str | object | None] = ...) -> None: ...
def trace_structured(name: str, metadata_fn: Callable[[], dict[str, Any] | tuple[str, int]] = ..., *, payload_fn: Callable[[], str | object | None] = ..., suppress_context: bool = False, expect_trace_id: bool = True, record_logging_overhead: bool = True, compile_id: CompileId | None = None) -> None:
    """
    metadata is an arbitrary JSON compatible struct, but it's expected to not be
    too long (e.g., less than 1MB)

    payload is an arbitrary string, which can be arbitrarily long (but expected to have
    newlines so no lines are too long)
    """
def dtrace_structured(name: str, metadata_fn: Callable[[], dict[str, Any] | tuple[str, int]] = ..., *, payload_fn: Callable[[], str | object | None] = ..., suppress_context: bool = False, expect_trace_id: bool = False, record_logging_overhead: bool = True) -> None:
    """
    For logging more detailed information used for debugging. This may result in
    the program becoming slow.
    """
