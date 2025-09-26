import contextlib
import torch
from _typeshed import Incomplete
from collections import Counter
from collections.abc import Generator
from torch._dynamo.external_utils import FakeCompiledAutogradEngine as FakeCompiledAutogradEngine, call_accumulate_grad as call_accumulate_grad, call_backward as call_backward, call_hook as call_hook, unwrap_maybe_dynamic_int as unwrap_maybe_dynamic_int
from torch._dynamo.source import GetItemSource as GetItemSource, LocalSource as LocalSource
from torch._dynamo.utils import counters as counters, get_chromium_event_logger as get_chromium_event_logger, lazy_format_graph_code as lazy_format_graph_code, set_locals_to_steal as set_locals_to_steal
from torch._functorch._aot_autograd.runtime_wrappers import AutogradLazyBackwardCompileInfo as AutogradLazyBackwardCompileInfo, CachedAutogradLazyBackwardCompileInfo as CachedAutogradLazyBackwardCompileInfo
from torch._guards import CompileContext as CompileContext, CompileId as CompileId, compile_context as compile_context
from torch._logging import getArtifactLogger as getArtifactLogger, trace_structured as trace_structured
from torch._prims_common import clone_preserve_strides as clone_preserve_strides
from torch._subclasses import FakeTensorMode as FakeTensorMode
from torch.fx import GraphModule as GraphModule
from torch.fx.experimental._backward_state import BackwardState as BackwardState
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, PythonKeyTracer as PythonKeyTracer, decompose as decompose, disable_autocast_cache as disable_autocast_cache, disable_proxy_modes_tracing as disable_proxy_modes_tracing, fetch_object_proxy as fetch_object_proxy, track_tensor_tree as track_tensor_tree
from torch.fx.experimental.symbolic_shapes import DimDynamic as DimDynamic, ShapeEnv as ShapeEnv
from torch.fx.proxy import Proxy as Proxy
from torch.fx.traceback import preserve_node_meta as preserve_node_meta, set_stack_trace as set_stack_trace
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._traceback import CapturedTraceback as CapturedTraceback

TURN_OFF_MSG: str
compiled_autograd_log: Incomplete
verbose_log: Incomplete

def snapshot_verbose_logging_enabled(): ...
def snapshot_cudagraph_enabled(): ...
def maybe_clone(x): ...
def extract_bw_module(CompiledFunction): ...

class NaNChecker:
    accumulate_grad: Incomplete
    params_indices: list[int]
    params_to_check: dict[str, torch.Tensor]
    output_names: list[str]
    def __init__(self, accumulate_grad: bool) -> None: ...
    def prep_with_graph(self, graph: torch.fx.Graph): ...
    def prep_with_inputs(self, inputs: tuple[torch.Tensor]): ...
    def check(self, out: tuple[torch.Tensor]): ...

class OpNamespace:
    custom_function_name_counter: Counter[str]
    def __init__(self) -> None: ...
    def add(self, name, fn, is_custom_function, is_traceable): ...
    def get(self, name): ...

class Op:
    fn: Incomplete
    is_custom_function: Incomplete
    __name__: Incomplete
    __module__: str
    def __init__(self, name, fn, is_custom_function) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __repr__(self) -> str: ...

ops: Incomplete
_graph_placeholders: Incomplete
_impure_targets: Incomplete
COMPILE_COUNTER: Incomplete

def make_compile_context(compiled_autograd_id): ...

class AutogradCompilerInstance:
    compiler_fn: Incomplete
    stack: Incomplete
    close: Incomplete
    shape_env: Incomplete
    fake_tensor_mode: Incomplete
    fx_tracer: Incomplete
    proxy_mode: Incomplete
    hooks_proxy: Proxy | None
    def __init__(self, compiler_fn) -> None: ...
    def wrap_fake(self, x, source): ...
    @staticmethod
    def source(name, idx) -> GetItemSource: ...
    id: Incomplete
    aot_id_counter: dict[int, int]
    compile_context: Incomplete
    nan_checker: Incomplete
    start_time_ns: Incomplete
    symnode_proxy_lookup: Incomplete
    def begin_capture(self, inputs: list[torch.Tensor], sizes: list[int], scalars: list[int | float], origins: list[list[tuple[int, str]]], accumulate_grad: bool, check_nans: bool): ...
    def log_compile_reasons(self, compile_reasons: list[str]): ...
    def proxy_call_aot_backward(self, pinputs, psaved_tensors, saved_tensors, pctx, ctx, maybe_backward_state_idx): ...
    def proxy_call_backward(self, inputs, output_metadatas, saved_tensors, backward_idx: int, ctx: torch.autograd.function.BackwardCFunction, maybe_backward_state_idx: int | None): ...
    def call_copy_slices_prologue(self, inputs, base_sizes, base_strides, base_storage_offset, view_sizes, view_strides, view_storage_offset): ...
    def call_copy_slices_epilogue(self, needs_input_grad, result, res, grad_slice): ...
    def allocate_dummy(self): ...
    def bind_function(self, fn_name, fn, is_custom_function, is_traceable):
        """Binds ops.fn_name = fn"""
    def apply_functional(self, fn_name, grads, args, output_metadata):
        """Proxies a call to ops.fn_name(grads, *args) into the graph"""
    def proxy_call(self, fn, args, output_metadata):
        """Proxies a call to fn(*args) into the graph"""
    def validate_outputs(self, _, outputs, args, output_metadata):
        """Proxies a call to ops.validate_outputs(outputs, *args) into the graph"""
    def accumulate(self, old_var, new_var): ...
    def accumulate_grad(self, variable, grad, has_post_hooks) -> None: ...
    def proxy_call_hook(self, hook, *args, **kwargs): ...
    def unpack_hook(self, hook_id, data_id): ...
    def tensor_pre_hook(self, inputs, hook_id, i: int): ...
    def cpp_tensor_pre_hook(self, inputs: list[torch.Tensor], hook_id: int, i: int): ...
    def pre_hook(self, inputs, hook_id): ...
    def post_hook(self, outputs, inputs, hook_id): ...
    def post_acc_grad_hook(self, input, hook_id): ...
    def move_graph_nodes_to_cuda(self, graph) -> list[int]: ...
    def is_sym_node(self, node): ...
    def dce(self): ...
    def remove_unused_sizes(self): ...
    def create_graph_module(self, id): ...
    def end_capture(self, outputs): ...
    @staticmethod
    def get_all_nodes(args): ...
    @staticmethod
    def is_placeholder(node): ...
    def reorder_accumulate_grad_nodes(self) -> None:
        """
        Usage of AOTAutograd causes all the accumulate_grad_ nodes to get pushed to the end of
        the graph.  This differs from eager mode, which schedules them as soon as possible. This
        pass attempts to reorder the graph to mimic eager behavior.
        """
    def delay_unpack_hook_nodes(self) -> None:
        """
        We can delay unpack hooks until they are needed, even later than in the eager autograd engine.
        """
    def reorder_tensor_pre_hook_nodes(self) -> None:
        """
        Usage of AOTAutograd causes all the tensor_pre_hook nodes to get pushed
        to the end of the graph. This differs from eager mode, which schedules
        them as soon as possible. This pass attempts to reorder the graph to
        mimic eager behavior.
        """
    def reorder_pre_hook_nodes_to_schedule_asap(self) -> None:
        """
        In this function, we schedule the pre hooks as soon as possible. This
        does not match eager behavior (schedule pre hook right before its
        registered node), but it can make acc grad be scheduled properly when
        the pre hooks are registered to them. After reordering acc grad node, we
        will reorder the pre hooks again to mimic eager behavior.
        """
    def reorder_pre_hook_nodes_to_mimic_eager(self) -> None:
        """
        Usage of AOTAutograd causes all the pre_hook nodes to get pushed to the
        end of the graph. This differs from eager mode, which schedules them
        right before their registered node execution. This pass attempts to
        reorder the graph to mimic eager behavior.
        """
    def reorder_post_acc_grad_hook_nodes(self) -> None:
        """
        Usage of AOTAutograd causes all the post_acc_grad_hook nodes to get
        pushed to the end of the graph. This differs from eager mode, which
        schedules them as soon as possible. This pass attempts to reorder the
        graph to mimic eager behavior.
        """
    def reorder_post_hook_nodes(self) -> None:
        """
        Usage of AOTAutograd causes all the post_hook nodes to get pushed to the
        end of the graph. This differs from eager mode, which schedules them as
        soon as possible. This pass attempts to reorder the graph to mimic eager
        behavior.
        """
    def to_proxy(self, t): ...
    def bind_objects_to_proxies(self, objects, proxies, origins: list[tuple[int, str]] | None = None): ...
    def bind_backward_state(self, index: int): ...
    def set_node_origin(self, node_name: str, nodecall_index: int, pyobj: torch.autograd.Function | None): ...

compiled_autograd_enabled: bool
compiled_autograd_enabled_force_eager: bool
in_compiled_autograd_region: bool
active_disable_ctx: bool
depth: int

@contextlib.contextmanager
def _enable(compiler_fn, dynamic: bool = True, ignore_active_disable_ctx: bool = True): ...
@contextlib.contextmanager
def _disable() -> Generator[None]: ...
def reset() -> None: ...
def copy_slices_prologue(inputs, base_sizes, base_strides, base_storage_offset, view_sizes, view_strides, view_storage_offset): ...
def copy_slices_epilogue(needs_input_grad, result, res, grad_slice): ...
