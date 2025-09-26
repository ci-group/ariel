import atexit
import contextlib
import functools
import threading
import torch
import torch.fx
import types
from . import config as config, convert_frame as convert_frame, distributed as distributed, external_utils as external_utils, trace_rules as trace_rules, utils as utils
from .backends.registry import CompilerFn as CompilerFn, lookup_backend as lookup_backend
from .code_context import code_context as code_context
from .exc import CondOpArgsMismatchError as CondOpArgsMismatchError, ShortenTraceback as ShortenTraceback, Unsupported as Unsupported, UserError as UserError, UserErrorType as UserErrorType
from .hooks import Hooks as Hooks
from .mutation_guard import install_generation_tagging_init as install_generation_tagging_init
from .types import CacheEntry as CacheEntry, DynamoCallback as DynamoCallback
from .utils import common_constant_types as common_constant_types, compile_times as compile_times
from _typeshed import Incomplete
from dataclasses import dataclass
from enum import Enum
from torch import _guards as _guards
from torch._C._dynamo.eval_frame import reset_code as reset_code, set_code_exec_strategy as set_code_exec_strategy, set_eval_frame as set_eval_frame, set_guard_complete_hook as set_guard_complete_hook, set_guard_error_hook as set_guard_error_hook, set_skip_guard_eval_unsafe as set_skip_guard_eval_unsafe, unsupported as unsupported
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._dynamo.types import ConvertFrameReturn as ConvertFrameReturn, FrameAction as FrameAction, FrameExecStrategy as FrameExecStrategy
from torch._export.utils import _compiling_state_context as _compiling_state_context
from torch._subclasses import fake_tensor as fake_tensor
from torch._subclasses.fake_tensor import unset_fake_temporarily as unset_fake_temporarily
from torch._utils_internal import justknobs_check as justknobs_check, log_export_usage as log_export_usage
from torch.export.dynamic_shapes import Constraint as Constraint, _DimHint as _DimHint, _DimHintType as _DimHintType, _IntWrapper as _IntWrapper, _RelaxedConstraint as _RelaxedConstraint, _combine_args as _combine_args, _process_dynamic_shapes as _process_dynamic_shapes
from torch.fx import GraphModule as GraphModule
from torch.fx.experimental._dynamism import clone_and_convert_to_meta as clone_and_convert_to_meta, track_dynamism_across_examples as track_dynamism_across_examples
from torch.fx.experimental.proxy_tensor import make_fx as make_fx
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError as ConstraintViolationError, DimDynamic as DimDynamic, ShapeEnv as ShapeEnv, StatelessSymbolicContext as StatelessSymbolicContext
from torch.fx.graph import _PyTreeCodeGen as _PyTreeCodeGen, _PyTreeInfo as _PyTreeInfo
from typing import Any, Callable, NamedTuple

log: Incomplete
always_optimize_code_objects: Incomplete
null_context = contextlib.nullcontext

class Unset(Enum):
    token = 0

cached_backends: dict[int, CompilerFn]
unset: Incomplete

def _maybe_set_eval_frame(callback: DynamoCallback): ...

@dataclass
class DynamoStance:
    stance: str = ...
    skip_guard_eval_unsafe: bool = ...
    backend: str | Callable[..., Any] | None = ...

_stance: Incomplete

def _set_stance(stance: DynamoStance) -> DynamoStance: ...

_EXAMPLE_INPUTS: dict[str, list[Any]] | None

def get_example_inputs(key) -> list[Any]: ...
def _callback_from_stance(callback): ...
def _create_wrapped_callback(compiler_fn): ...
def _get_or_add_example_inputs(frame): ...
def _create_delayed_compile_callback(callback, stance): ...
def _is_skip_guard_eval_unsafe_stance(): ...
def _reset_guarded_backend_cache() -> None: ...

DONT_WRAP_FILES: Incomplete

def _debug_get_cache_entry_list(code: types.CodeType | Callable[..., Any]) -> list[CacheEntry]:
    """
    Given a code object or a callable object, retrieve the cache entries
     stored in this code.
    """

class OptimizedModule(torch.nn.Module):
    """
    Wraps the original nn.Module object and later patches its
    forward method to optimized self.forward method.
    """
    _torchdynamo_orig_callable: Callable[..., Any]
    get_compiler_config: Callable[[], Any]
    _opt_mod_attributes: Incomplete
    _super_module_initialized: bool
    _orig_mod: Incomplete
    dynamo_ctx: Incomplete
    def __init__(self, mod: torch.nn.Module, dynamo_ctx) -> None: ...
    forward: Incomplete
    _forward: Incomplete
    def _initialize(self) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __reduce__(self): ...
    def __getstate__(self): ...
    __dict__: Incomplete
    def __setstate__(self, state) -> None: ...
    @property
    def training(self): ...
    @training.setter
    def training(self, value) -> None: ...
    def __getattr__(self, name): ...
    def __setattr__(self, name, val) -> None: ...
    def __delattr__(self, name): ...
    def _call_lazy_check(self, *args, **kwargs): ...
    def __dir__(self): ...

def remove_from_cache(f) -> None:
    """
    Make sure f.__code__ is not cached to force a recompile
    """
def nothing() -> None: ...
def always_false(): ...
def innermost_fn(fn):
    """
    In case of nesting of _TorchDynamoContext calls, find the innermost
    function. TorchDynamo caches on fn.__code__ object, so its necessary to find
    the innermost function to pass on the optimize, run, disable etc.
    """
def make_set_enable_dynamic(enable: bool): ...

class DynamoTLS(threading.local):
    traced_frame_infos: list[str]

dynamo_tls: Incomplete

def clear_dynamo_tls() -> None: ...
@atexit.register
def _log_traced_frames() -> None:
    """
    At program exit, log all of the frames Dynamo has attempted to trace from,
    excluding the continuation frames generated by Dynamo.
    """
def guard_collectives_hook(guard_eval_result): ...

_not_set: Incomplete

class _TorchDynamoContext:
    callback: DynamoCallback
    _backend_ctx_ctor: Incomplete
    prior: Unset | DynamoCallback
    first_ctx: Incomplete
    export: Incomplete
    _dynamic: Incomplete
    compiler_config: Incomplete
    cleanup_fns: list[Callable[[], Any]]
    enter_exit_hooks: Incomplete
    _package: Incomplete
    def __init__(self, callback: DynamoCallback, on_enter=..., backend_ctx_ctor=..., patch_fn=..., first_ctx: bool = False, *, export: bool = False, dynamic=None, compiler_config=None, package=None) -> None: ...
    prior_skip_guard_eval_unsafe: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def __call__(self, fn): ...

class OptimizeContext(_TorchDynamoContext):
    def __init__(self, callback, backend_ctx_ctor, first_ctx: bool = False, *, export: bool = False, dynamic=None, compiler_config=None, rebuild_ctx: Callable[[], OptimizeContext | _NullDecorator] | None = None, package=None) -> None: ...
    def __reduce__(self): ...

class RunOnlyContext(_TorchDynamoContext):
    def __init__(self) -> None: ...
    def __reduce__(self): ...

class DisableContext(_TorchDynamoContext):
    msg: Incomplete
    wrapping: Incomplete
    def __init__(self, msg: str | None = None, wrapping: bool = True) -> None: ...
    def __call__(self, fn): ...
    def __reduce__(self): ...

def _optimize_catch_errors(compile_fn, hooks: Hooks, backend_ctx_ctor=..., export: bool = False, dynamic=None, compiler_config=None, rebuild_ctx=None, package=None): ...
def get_compiler_fn(compiler_fn): ...

class _NullDecorator(contextlib.nullcontext):
    def __call__(self, fn): ...

def check_if_dynamo_supported() -> None: ...
def is_dynamo_supported(): ...
def check_if_inductor_supported() -> None: ...
def is_inductor_supported(): ...
def check_for_incompatible_configs() -> None: ...
def optimize(*args, **kwargs): ...
def _optimize(rebuild_ctx: Callable[[], OptimizeContext | _NullDecorator], backend: str = 'inductor', *, nopython: bool = False, guard_export_fn=None, guard_fail_fn=None, guard_filter_fn=None, disable: bool = False, dynamic=None, package=None) -> OptimizeContext | _NullDecorator:
    '''
    The main entrypoint of TorchDynamo.  Do graph capture and call
    backend() to optimize extracted graphs.

    Args:
        backend: One of the two things:
            - Either, a function/callable taking a torch.fx.GraphModule and
            example_inputs and returning a python callable that runs the
            graph faster.
            One can also provide additional context for the backend, like
            torch.jit.fuser("fuser2"), by setting the backend_ctx_ctor attribute.
            See AOTAutogradMemoryEfficientFusionWithContext for the usage.
            - Or, a string backend name in `torch._dynamo.list_backends()`
        nopython: If True, graph breaks will be errors and there will
            be a single whole-program graph.
        disable: If True, turn this decorator into a no-op
        dynamic: If True, upfront compile as dynamic a kernel as possible.  If False,
            disable all dynamic shapes support (always specialize).  If None, automatically
            detect when sizes vary and generate dynamic kernels upon recompile.

    Example Usage::

        @torch._dynamo.optimize()
        def toy_example(a, b): ...
    '''
def explain(f, *extra_args, **extra_kwargs): ...

class FlattenInputOutputSignature(torch.fx.Transformer):
    new_args: Incomplete
    old_args_gen: Incomplete
    matched_output_elements_positions: Incomplete
    flat_results: Incomplete
    def __init__(self, m: torch.fx.GraphModule, flat_args: tuple[Any], matched_input_elements_positions: list[int], flat_results: list[Any], matched_output_elements_positions: list[int], example_fake_inputs: list[torch.Tensor], flat_args_dynamic_dims: list[set[int]], fake_mode: fake_tensor.FakeTensorMode | None = None) -> None: ...
    def placeholder(self, target, args, kwargs): ...
    def output(self, target, args, kwargs): ...
    current_node: Incomplete
    def run_node(self, n): ...
    def transform(self): ...

class ExportResult(NamedTuple):
    graph_module: torch.fx.GraphModule
    guards: _guards.GuardsSet

def check_signature_rewritable(graph) -> None: ...
def rewrite_signature(f_sig, graph, fake_mode, flat_args, in_spec, example_fake_inputs, graph_captured_input, graph_captured_output, dynamo_traced_result, flat_args_dynamic_dims): ...
def export(f: Callable[..., Any], *extra_args, aten_graph: bool = False, pre_dispatch: bool = False, decomposition_table: dict[torch._ops.OpOverload, Callable[..., Any]] | None = None, tracing_mode: str = 'symbolic', dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None, specialize_float: bool = True, assume_static_by_default: bool = False, same_signature: bool = True, disable_constraint_solver: bool = False, prefer_deferred_runtime_asserts_over_guards: bool = False, allow_complex_guards_as_runtime_asserts: bool = False, _log_export_usage: bool = True, constraints: list[Constraint] | None = None, **extra_kwargs) -> Callable[..., ExportResult]:
    '''
    Export an input function f to a format that can be executed outside of PyTorch using the FX graph.

    Args:
        f (callable): A PyTorch function to be exported.

        aten_graph (bool): If True, exports a graph with ATen operators.
        If False, exports a graph with Python operators. Default is False.

        pre_dispatch (bool): If True, exports a graph with ATen operators,
        but before any logic in the PyTorch dispatcher has run.
        This can be useful if you want to apply further transformations on a graph before running it
        through autograd, autocast, or any other functionalities that are integrated into the dispatcher.
        This flag is only valid if aten_graph=True is set.
        Default is False.

        decomposition_table (dict): A dictionary that maps operators to their decomposition functions.
        Required if aten_graph or tracing_mode is specified. Default is None.

        tracing_mode (str): If "symbolic", turn on dynamic shapes support. Default is "symbolic".

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

        same_signature (bool): If True, rewrite the returned graph\'s signature to be the same as f.

        disable_constraint_solver (bool): Whether the dim constraint solver must be disabled.

    Returns:
        A function that given args and kwargs, returns a tuple of (graph, guards)
        Graph: An FX graph representing the execution of the input PyTorch function with the provided arguments and options.
        Guards: The guards we accumulated during tracing f above

    Raises:
        AssertionError: If decomposition_table is specified without setting aten_graph=True,
        or if graph breaks during tracing in export.

        AssertionError: If Dynamo input and output is not consistent with traced input/output.

    Note - this headerdoc was authored by ChatGPT, with slight modifications by the author.
    '''
def optimize_assert(*args, **kwargs): ...
def _optimize_assert(rebuild_ctx: Callable[[], OptimizeContext], backend, *, hooks=..., export: bool = False, export_constraints=None, dynamic=None, package=None):
    """
    The same as `torch._dynamo.optimize(backend, nopython=True)`
    """

class TorchPatcher:
    @staticmethod
    @functools.cache
    def patch() -> None: ...
    @staticmethod
    def suppress_torch_distributed_warnings(fn): ...

def skip_code(code: types.CodeType): ...
