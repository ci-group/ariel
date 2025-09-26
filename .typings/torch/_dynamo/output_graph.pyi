import contextlib
import functools
import sympy
import torch
import torch._guards
import torch.nn
import traceback
from . import config as config, exc as exc, variables as variables
from .backends.registry import CompiledFn as CompiledFn, CompilerFn as CompilerFn
from .bytecode_transformation import Instruction as Instruction, create_call_function as create_call_function, create_instruction as create_instruction, create_load_const as create_load_const, unique_id as unique_id
from .code_context import code_context as code_context
from .codegen import PyCodegen as PyCodegen
from .current_scope_id import enter_new_scope as enter_new_scope
from .device_interface import get_interface_for_device as get_interface_for_device
from .exc import BackendCompilerFailed as BackendCompilerFailed, SkipFrame as SkipFrame, exceptions_allowed_to_be_fallback as exceptions_allowed_to_be_fallback, unimplemented_v2 as unimplemented_v2, unimplemented_v2_with_warning as unimplemented_v2_with_warning
from .graph_deduplication import apply_graph_deduplication as apply_graph_deduplication
from .graph_region_tracker import GraphRegionTracker as GraphRegionTracker
from .guards import GuardBuilder as GuardBuilder, install_guard as install_guard
from .mutation_guard import is_dynamic_nn_module as is_dynamic_nn_module
from .side_effects import AttributeMutationExisting as AttributeMutationExisting, SideEffects as SideEffects
from .source import AttrSource as AttrSource, BackwardStateSource as BackwardStateSource, ConstantSource as ConstantSource, GetItemSource as GetItemSource, GlobalStateSource as GlobalStateSource, LocalSource as LocalSource, NumpyTensorSource as NumpyTensorSource, ParamBufferSource as ParamBufferSource, ShapeEnvSource as ShapeEnvSource, SyntheticLocalSource as SyntheticLocalSource, TensorProperty as TensorProperty, TensorPropertySource as TensorPropertySource, is_constant_source as is_constant_source, is_from_local_source as is_from_local_source
from .utils import CleanupHook as CleanupHook, LazyString as LazyString, _extract_tensor_dict as _extract_tensor_dict, checkpoint_params as checkpoint_params, clone_inputs as clone_inputs, count_calls as count_calls, counters as counters, dynamo_timed as dynamo_timed, get_instruction_source_311 as get_instruction_source_311, get_locals_to_steal as get_locals_to_steal, get_static_address_type as get_static_address_type, get_unique_name_wrt as get_unique_name_wrt, graph_break_reasons as graph_break_reasons, increment_op_count as increment_op_count, istype as istype, lazy_format_graph_code as lazy_format_graph_code, nn_module_proxy as nn_module_proxy, same as same, set_example_value as set_example_value
from .variables.base import VariableTracker as VariableTracker
from .variables.builder import BackwardStateGraphArg as BackwardStateGraphArg, GraphArg as GraphArg, TrackedFake as TrackedFake, wrap_fx_proxy as wrap_fx_proxy
from .variables.ctx_manager import ContextWrappingVariable as ContextWrappingVariable
from .variables.lists import BaseListVariable as BaseListVariable
from .variables.misc import CellVariable as CellVariable, NullVariable as NullVariable
from .variables.nn_module import NNModuleVariable as NNModuleVariable
from .variables.tensor import NumpyNdarrayVariable as NumpyNdarrayVariable, SymNodeVariable as SymNodeVariable, TensorVariable as TensorVariable, UnspecializedPythonVariable as UnspecializedPythonVariable
from .variables.torch_function import TensorWithTFOverrideVariable as TensorWithTFOverrideVariable
from _typeshed import Incomplete
from collections.abc import Generator
from dataclasses import dataclass, field as dc_field
from torch import Tensor as Tensor, fx as fx
from torch._C._dynamo import guards as guards
from torch._dynamo.exc import ShortenTraceback as ShortenTraceback, TensorifyScalarRestartAnalysis as TensorifyScalarRestartAnalysis
from torch._dynamo.symbolic_convert import InstructionTranslatorBase as InstructionTranslatorBase
from torch._guards import CompileContext as CompileContext, CompileId as CompileId, GlobalContextCheckpointState as GlobalContextCheckpointState, Source as Source, TracingContext as TracingContext, tracing as tracing
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor
from torch._utils_internal import signpost_event as signpost_event
from torch.fx._lazy_graph_module import _make_graph_module as _make_graph_module
from torch.fx.experimental._backward_state import BackwardState as BackwardState
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv, Specialization as Specialization, free_symbols as free_symbols, guard_scalar as guard_scalar, is_symbolic as is_symbolic
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts as insert_deferred_runtime_asserts
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass
from typing import Any, Callable

log: Incomplete
graph_tabular_log: Incomplete
graph_code_log: Incomplete
graph_sizes_log: Incomplete
trace_call_log: Incomplete
RootGuardManager = guards.RootGuardManager

@dataclass(frozen=True)
class VariableTrackerCacheKey:
    vt_id: int
    source: Source

@dataclass(frozen=True)
class AliasingInfo:
    has_aliasing: bool
    msg: str

@dataclass(frozen=True)
class MutationInfo:
    has_mutation: bool
    msg: str

class VariableTrackerCache:
    cache: Incomplete
    def __init__(self) -> None: ...
    def lookup(self, value, source): ...
    def add(self, value, source, vt) -> None: ...
    def clone(self): ...
    def clear(self) -> None: ...

@functools.cache
def _step_logger(): ...

@dataclass
class GraphCompileReason:
    """Stores why a given output graph was compiled; i.e. what caused the graph break."""
    reason: str
    user_stack: list[traceback.FrameSummary]
    graph_break: bool = ...
    def __post_init__(self) -> None: ...

def _get_gen_rand_values_fn(random_calls): ...

class FakeRootModule(torch.nn.Module):
    """Trick the constructor of fx.GraphModule"""
    def __init__(self, nn_modules: dict[str, torch.nn.Module]) -> None: ...
    def __repr__(self) -> str: ...
    def add_nn_modules(self, nn_modules: dict[str, torch.nn.Module]): ...

class WrapperBackend:
    backend: CompilerFn
    def __init__(self, backend: CompilerFn) -> None: ...
    restore: Incomplete
    gm: Incomplete
    candidate: Incomplete
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]): ...
Scope = dict[str, object]

@dataclass
class OutputGraphGuardsState:
    '''
    A base class containing fields that are considered "persistent" when we
    want to save all the important state for reconstrucing guards in a different
    process. Normally we don\'t need to add states here, but we may have to when
    the information is needed to serialize the guards, so the fields here are
    supposed to be serializable as a requirement.
    '''
    local_scope: Scope
    global_scope: Scope
    torch_function_mode_stack: list[torch.overrides.TorchFunctionMode]
    guard_on_key_order: set[Source]
    input_source_to_sizes_strides: dict[Source, dict[str, Any]]
    dual_level: int
    functorch_layers: list[torch._functorch.pyfunctorch.FuncTorchInterpreter]
    current_device: torch.device | None
    export: bool = ...
    export_constraints: bool = ...
    _guards: torch._guards.GuardsSet | None = ...
    _aotautograd_guards: list[torch._guards.GuardEnvExpr] | None = ...
    @property
    def shape_env(self) -> None: ...
    @property
    def guards(self): ...
    @property
    def aotautograd_guards(self): ...

@dataclass
class StackLocalsMetadata:
    """
    Stores metadata for a frame's stack and locals for the purposes of building resume functions
    """
    stack_null_idxes: list[int] = dc_field(default_factory=list)
    locals_null_keys: list[str] = dc_field(default_factory=list)
    stack_ctx_args: list[tuple[int, tuple[Any, ...]]] = dc_field(default_factory=list)
    stack_ctx_idxes_orig: list[int] = dc_field(default_factory=list)
    locals_ctx_args: list[tuple[str, tuple[Any, ...]]] = dc_field(default_factory=list)

class OutputGraph(OutputGraphGuardsState):
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.

    OutputGraph is 1:1 with a frame being processed. Each frame is associated
    with some root InstructionTranslator. When user code calls a function,
    we construct a InliningInstructionTranslator that continues to write into
    the root InstructionTranslator's OutputGraph.
    """
    side_effects: SideEffects
    tracers: Incomplete
    input_source_to_var: dict[Source, VariableTracker]
    export: Incomplete
    export_constraints: Incomplete
    frame_state: Incomplete
    cleanup_hooks: list[Callable[[], Any]]
    compile_id: int
    installed_globals: set[str]
    co_fields: Incomplete
    region_tracker: Incomplete
    tracked_fakes: list[TrackedFake]
    tracing_context: TracingContext
    dynamo_compile_id: CompileId | None
    tracked_fakes_id_to_source: dict[int, list[Source]]
    param_name_to_source: dict[str, Source] | None
    variable_tracker_cache: Incomplete
    unique_var_id: Incomplete
    code_options: dict[str, Any]
    output_instructions: list[Instruction]
    timestamp: int
    register_finalizer_fns: list[Callable[[fx.GraphModule], None]]
    compiler_fn: CompilerFn | None
    root_tx: Incomplete
    package: Incomplete
    source_to_user_stacks: dict[Source, list[traceback.StackSummary]]
    _current_tx: list[InstructionTranslatorBase]
    cleanups: list[CleanupHook]
    should_exit: bool
    unspec_variable_map: dict[str, UnspecializedPythonVariable]
    torch_function_mode_enabled: Incomplete
    has_user_defined_allowed_in_graph: bool
    non_compliant_ops: set[torch._ops.OpOverload]
    compliant_custom_ops: set[torch._ops.OpOverload]
    dynamo_flat_name_to_original_fqn: dict[str, str]
    random_calls: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]]
    random_values_var: Any
    pregraph_bytecode: list[Instruction]
    backward_state: dict[str, VariableTracker]
    backward_state_proxy: torch.fx.Proxy | None
    backward_state_var: str | None
    name_of_builtins_dict_key_in_fglobals: str
    compiler_trace_stack: Incomplete
    saved_tensors_hooks_subgraph_names: list[str] | None
    def __init__(self, code_options: dict[str, Any], compiler_fn: CompilerFn | None, root_tx, export: bool, export_constraints, frame_state, local_scope: Scope, global_scope: Scope, f_code, torch_function_mode_stack, package) -> None: ...
    def mark_bytecode_tracing_start(self) -> None: ...
    def mark_bytecode_tracing_stop(self) -> None: ...
    def install_builtins_dict_in_fglobals(self): ...
    def add_backward_state_hook(self, hook: VariableTracker, prefix: str = 'hook'): ...
    def get_backward_state_proxy(self): ...
    def init_ambient_guards(self) -> None: ...
    def maybe_install_saved_tensors_hooks_subgraphs(self) -> list[str] | None: ...
    def dump_guards_state(self): ...
    def synthetic_graph_input(self, fn, args):
        """
        call fn(*args) before the graph runs and turn the result into a fake input.
        """
    def add_cleanup_hook(self, fn: Callable[[], Any]): ...
    def call_cleanup_hooks(self) -> None: ...
    @property
    def root_tracer(self): ...
    @property
    def current_tracer(self): ...
    def is_root_tracer(self): ...
    @property
    def graph(self): ...
    @graph.setter
    def graph(self, value) -> None: ...
    @property
    def input_name_to_proxy(self): ...
    @property
    def real_value_cache(self): ...
    @property
    def bound_symbols(self): ...
    def create_proxy(self, *args, **kwargs): ...
    def create_node(self, *args, **kwargs): ...
    def remove_node(self, *args, **kwargs): ...
    @contextlib.contextmanager
    def subtracer(self, source_target, prior_tracer) -> Generator[Incomplete]: ...
    @property
    def output(self): ...
    @property
    def fake_mode(self): ...
    @property
    def shape_env(self): ...
    @property
    def guards(self) -> torch._guards.GuardsSet: ...
    @property
    def nn_modules(self) -> dict[str, Any]: ...
    @property
    def aotautograd_guards(self): ...
    def save_global_state(self, out=None) -> None:
        """
        Saves to out if it is provided. Else saves to the tracing context's global_state.
        """
    def push_tx(self, tx) -> None: ...
    def pop_tx(self): ...
    @property
    def current_tx(self): ...
    def count_calls(self): ...
    def is_empty_graph(self): ...
    def get_submodule(self, keys): ...
    def new_var(self, name: str = 'tmp'): ...
    def update_co_names(self, name) -> None:
        """Ensure self.code_options.co_names contains name"""
    @staticmethod
    def module_key_name(*names): ...
    def register_static_attr_and_return_proxy(self, attr_prefix: str, attr_value: Any) -> fx.Proxy: ...
    def register_attr_or_module(self, target: torch.nn.Module | torch.Tensor | Any, *names, **options): ...
    def handle_aliases_for_stolen_lists(self, tx): ...
    def _get_stack_values_to_restore(self, tx, stack_pops):
        """
        Gets the stack + locals values belonging to tx that need to be restored.

        Also prunes dead tx locals and realizes all VTs in the tx's stack.

        NullVariables in stack/locals will NOT be restored, unless they are the top `stack_pops`
        elements of the stack - it is expected that the next instruction to run will pop the top
        `stack_pops` elements of the stack, so we should codegen NULLs.

        Returns:
            - stack_values: stack and locals values that need to be restored
            - restore_vars: names of locals corresponding to the locals part of `stack_values`
            - meta: locations of NULLs and ContextWrappingVariables in the stack/locals
                (ignores the top `stack_pops` values on the stack)
        """
    partial_convert: Incomplete
    compile_subgraph_reason: Incomplete
    def compile_subgraph(self, tx: InstructionTranslatorBase, reason: GraphCompileReason, partial_convert: bool = False, stack_pops: int = 0):
        """
        Compiles the current subgraph, with inputs w.r.t. self.root_tx, and codegens:
            - Call the compiled subgraph
            - Apply side effects
            - Codegen stack and locals
            - Store the locals

        Python does not allow NULL to be an arg to a function, so we do not codegen NULLs on the stack,
        unless the value is one of the top `stack_pops` values on the stack (these values are expected to be
        popped immediately after this generated code. The prologue of the resume function is expected to restore
        any dropped NULLs.

        Returns stack indices and locals keys where we dropped NULLs, and where we found inactive context manager objects.
        """
    def codegen_suffix(self, tx, stack_values, cg): ...
    def cleanup_graph(self) -> None:
        '''
        Remove "creation_timestamp" from node meta

        Remove this pattern from the graph:
            torch._C._set_grad_enabled(False)
            torch._C._set_grad_enabled(True)
        '''
    def get_graph_sizes_structured(self): ...
    def get_graph_sizes(self, name: str): ...
    @contextlib.contextmanager
    def restore_global_state(self) -> Generator[None]:
        """
        Momentarily restores the global state to what it was prior to tracing the current output
        """
    def run_compiler_collective(self): ...
    def compile_and_call_fx_graph(self, tx, rv, root):
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.

        Code is generated w.r.t. self.root_tx.
        tx is only used for preserving GraphModule metadata
        """
    @property
    def placeholders(self) -> list[fx.Node]: ...
    @property
    def graphargs(self) -> list[GraphArg]: ...
    def call_user_compiler(self, gm: fx.GraphModule, example_inputs: list[Tensor]) -> CompiledFn: ...
    def _call_user_compiler(self, gm: fx.GraphModule, example_inputs: list[Tensor]) -> CompiledFn: ...
    def dedup_pass(self): ...
    def install_subgraph(self, name, sub_gm): ...
    def example_inputs(self) -> list[torch.Tensor]: ...
    def remove_unused_get_attr_nodes(self) -> None: ...
    def remove_unused_graphargs(self) -> None: ...
    def remove_tensorify_specialized_graphargs(self) -> None: ...
    def add_output_instructions(self, prefix: list[Instruction]) -> None:
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.
        """
    def install_global_unsafe(self, name, value) -> None:
        """
        WARNING: prefer the safer `install_global_by_id/install_global`.
        torch.compile instances should be independent of each other;
        one footgun is to have one instance depend on the existence of
        a global installed by another instance. This can happen if we mangle
        a global the same way across both instances.
        """
    def install_global_by_id(self, prefix, value) -> str:
        """
        Installs a global if it hasn't been installed already.
        This is determined by (prefix, id(value)) pair.

        Returns the name of the newly installed global.
        """
    def install_global(self, prefix, value) -> str:
        """
        Installs a global, generating a unique name for it.

        Returns the name of the newly installed global.
        """
    def cleanup(self) -> None: ...
    def add_graph_finalizer(self, register_finalizer: Callable[[fx.GraphModule], None]) -> None: ...
    def example_value_from_input_node(self, node: torch.fx.Node):
        """Extract the non-fake example tensor"""

err_epilogue: str

def check_pt2_compliant_op(output_graph, kind, target, args, kwargs) -> None: ...

_compile_id_counter: Incomplete

class LazyProxy:
    tracer: Incomplete
    fn: Incomplete
    args: Incomplete
    kwargs: Incomplete
    def __init__(self, tracer, fn, *args, **kwargs) -> None: ...
    def __call__(self): ...

class SubgraphTracer(fx.Tracer):
    """
    Holds an FX graph that is being traced. OutputGraph owns a SubgraphTracer
    and the separation of responsibilities is that SubgraphTracer is
    responsible for building the graph while OutputGraph is responsible for
    compiling and executing the graph.
    """
    output_graph: Incomplete
    graph: Incomplete
    is_export: Incomplete
    input_name_to_proxy: dict[str, fx.Proxy]
    real_value_cache: dict[fx.Node, torch.Tensor]
    parent: Incomplete
    source_target: Incomplete
    lifted_freevars: Incomplete
    bound_symbols: dict[sympy.Symbol, torch.fx.Proxy | LazyProxy]
    prev_inst: Incomplete
    under_activation_checkpoint: bool
    allow_side_effects_under_checkpoint: bool
    unsafe_allow_externally_visible_side_effects: bool
    is_reconstructing_generator: bool
    debug_level: int
    _cur_code: Incomplete
    _orig_gm_meta: Incomplete
    _orig_gm_lineno_map: Incomplete
    _orig_gm_firstlineno: Incomplete
    source_fn_stack: Incomplete
    _used_names: OrderedSet[str]
    _input_versions_at_beginning: list[int]
    def __init__(self, output_graph, parent=None, is_export: bool = False, source_target=None) -> None: ...
    def _maybe_preserve_original_meta(self, tx, node) -> None: ...
    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None): ...
    def create_node(self, op, target, args=None, kwargs=None, name=None, type_expr=None): ...
    def remove_node(self, node) -> None: ...
    def create_graph_input(self, name, type_expr, example_value, before: bool = False, source=None): ...
    def lift_tracked_freevar_to_input(self, proxy): ...
    def maybe_lift_tracked_freevar_to_input(self, arg):
        """
        If arg is a free variable, then lift it to be an input.
        Returns the new lifted arg (if arg was a freevar), else the
        original arg.
        """
    def track_unbacked_symbols(self, example_value, e_proxy: LazyProxy | torch.fx.Proxy): ...
    def _lift_basic_symbols(self, example_value: torch.SymInt | torch.Tensor, src: Source | None): ...
    def lookup_unbound_symbols(self, s: torch.SymInt) -> list[sympy.Symbol]: ...
    def has_input_mutation(self): ...
    def has_aliasing(self): ...
