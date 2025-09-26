import contextlib
import sympy
import torch
import torch.fx
from . import config as config, ir as ir, metrics as metrics
from .codegen.common import BackendFeature as BackendFeature, DeviceOpOverrides as DeviceOpOverrides, FileBackedGraphModule as FileBackedGraphModule, WorkspaceArg as WorkspaceArg, get_backend_features as get_backend_features, get_device_op_overrides as get_device_op_overrides, get_wrapper_codegen_for_device as get_wrapper_codegen_for_device, init_backend_registration as init_backend_registration
from .codegen.wrapper import PythonWrapperCodegen as PythonWrapperCodegen
from .exc import CppWrapperCodegenError as CppWrapperCodegenError, LoweringException as LoweringException, MissingOperatorWithDecomp as MissingOperatorWithDecomp, MissingOperatorWithoutDecomp as MissingOperatorWithoutDecomp
from .ir import Constant as Constant, DonatedBuffer as DonatedBuffer, FixedLayout as FixedLayout, GraphPartitionSignature as GraphPartitionSignature, InputBuffer as InputBuffer, Pointwise as Pointwise, Reduction as Reduction, StorageBox as StorageBox, TensorBox as TensorBox, TorchBindObject as TorchBindObject, get_device_type as get_device_type
from .lowering import FALLBACK_ALLOW_LIST as FALLBACK_ALLOW_LIST, constrain_to_fake_tensors as constrain_to_fake_tensors, constrain_to_fx_strides as constrain_to_fx_strides, fallback_handler as fallback_handler, fallback_node_due_to_unsupported_type as fallback_node_due_to_unsupported_type, lowerings as lowerings, make_fallback as make_fallback, maybe_layout_constraints as maybe_layout_constraints, needs_realized_inputs as needs_realized_inputs, require_contiguous as require_contiguous, tag_to_layout_constraint as tag_to_layout_constraint, unsupported_output_tensor as unsupported_output_tensor
from .runtime import autotune_cache as autotune_cache
from .runtime.autotune_cache import AutotuneCacheBundler as AutotuneCacheBundler
from .scheduler import BaseSchedulerNode as BaseSchedulerNode
from .sizevars import SizeVarAllocator as SizeVarAllocator
from .utils import GraphPartitionMap as GraphPartitionMap, SUPPORTED_MKLDNN_DEVICES as SUPPORTED_MKLDNN_DEVICES, ValueWithLineMap as ValueWithLineMap, convert_shape_to_inductor as convert_shape_to_inductor, gather_origins as gather_origins, get_cloned_parameter_buffer_name as get_cloned_parameter_buffer_name, get_donated_idxs as get_donated_idxs, get_sympy_Expr_dtype as get_sympy_Expr_dtype, is_same_tensor as is_same_tensor, maybe_get_suppress_shape_guards_ctx as maybe_get_suppress_shape_guards_ctx, normalize_name as normalize_name, should_assume_input_aligned as should_assume_input_aligned
from .virtualized import NullHandler as NullHandler, V as V
from _typeshed import Incomplete
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from sympy import Expr as Expr
from torch import Tensor as Tensor, device as device
from torch._decomp import get_decompositions as get_decompositions
from torch._dynamo.utils import defake as defake, dynamo_timed as dynamo_timed
from torch._higher_order_ops.effects import _EffectType as _EffectType
from torch._inductor.codecache import output_code_log as output_code_log
from torch._inductor.fb.utils import log_module_code as log_module_code
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from torch._library.utils import get_layout_constraint_tag as get_layout_constraint_tag
from torch._logging import LazyString as LazyString, trace_structured as trace_structured
from torch._prims_common import compute_required_storage_length as compute_required_storage_length, make_channels_last_strides_for as make_channels_last_strides_for
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor
from torch._utils_internal import full_aoti_runtime_assert as full_aoti_runtime_assert
from torch.fx import GraphModule as GraphModule
from torch.fx.experimental._backward_state import BackwardState as BackwardState
from torch.fx.experimental.sym_node import magic_methods as magic_methods, method_to_operator as method_to_operator
from torch.fx.experimental.symbolic_shapes import RuntimeAssert as RuntimeAssert, ShapeEnv as ShapeEnv, SymTypes as SymTypes, SympyBoolean as SympyBoolean, _get_placeholder_expr as _get_placeholder_expr, free_unbacked_symbols as free_unbacked_symbols, has_free_symbols as has_free_symbols, resolve_unbacked_bindings as resolve_unbacked_bindings
from torch.fx.graph import Graph as Graph
from torch.fx.node import Node as Node
from torch.utils._mode_utils import no_dispatch as no_dispatch
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._sympy.numbers import int_oo as int_oo
from types import ModuleType
from typing import Any, Callable, NoReturn

CompiledModule = ModuleType | FileBackedGraphModule
log: Incomplete
perf_hint_log: Incomplete
aten: Incomplete
_post_grad_graph_counter: Incomplete

def may_get_constant_buffer_dtype(constant_buffer: sympy.Expr) -> torch.dtype | None: ...
def is_magic_method(op: Any) -> bool: ...
def getattr_recursive(obj: GraphModule, target: str) -> Tensor | torch._C.ScriptObject | GraphModule: ...
def get_user_visible_output_strides(g: Graph) -> dict[Node, tuple[int, ...]]: ...
def mark_nodes_dislike_padding(g: Graph, user_visible_output_strides: dict[Node, tuple[int, ...]]) -> None:
    """
    Nodes like convolution/convolution_backward want its input to be dense.
    If we pad their inputs, we result in extra calls to copy kernels!  On the other hand, padding usually helps reduction.

    The pass finds nodes that dislike padding. These are nodes that can be reached
    from a convolution/convolution_backward in the backward direction without
    going thru a reduction.
    """

class GraphLowering(torch.fx.Interpreter):
    graph_outputs: list[ir.IRNode]
    example_inputs: Incomplete
    layout_opt: Incomplete
    num_channels_last_conv: int
    is_inference: Incomplete
    is_backward: Incomplete
    is_const_graph: Incomplete
    const_wrapper_code: Incomplete
    const_kernel_code: Incomplete
    const_module: Incomplete
    inputs_to_check: Incomplete
    extra_traceback: bool
    reuse_shape_env: bool
    _shape_env: Incomplete
    ras_by_symbol: dict[sympy.Symbol | None, list[RuntimeAssert]]
    bound_unbacked_symbols: Incomplete
    sizevars: Incomplete
    graph_input_names: list[str]
    graph_inputs: dict[str, TensorBox | TorchBindObject | sympy.Expr]
    graph_inputs_original: dict[str, InputBuffer]
    partition_maps: list[GraphPartitionMap] | None
    zero_dim_cpu_tensor_list: OrderedSet[str]
    device_types: OrderedSet[str]
    device_idxs: OrderedSet[int]
    device_type: str
    buffer_to_padded_size: dict[str, list[int]]
    buffers: list[ir.Buffer]
    operations: list[ir.Operation]
    const_output_index: dict[str, int]
    folded_constants: OrderedSet[str]
    constants: dict[str, torch.Tensor]
    named_buffers: dict[str, torch.Tensor]
    named_parameters: dict[str, torch.Tensor]
    torchbind_constants: dict[str, torch._C.ScriptObject | FakeScriptObject]
    seen_subgraphs: dict[str, ir.Subgraph]
    constant_reprs: dict[str, str]
    removed_operations: OrderedSet[str]
    removed_buffers: OrderedSet[str]
    removed_inplace_buffers: OrderedSet[str]
    mutated_buffers: OrderedSet[str]
    never_reuse_buffers: OrderedSet[str]
    inplaced_to_remove: OrderedSet[str]
    device_ops: DeviceOpOverrides
    wrapper_code: PythonWrapperCodegen
    extern_kernel_nodes: list[ir.ExternKernelNode]
    extern_node_serializer: Callable[[list[ir.ExternKernelNode]], Any]
    current_node: torch.fx.Node
    lists: dict[str, list[str]]
    mutated_inputs: OrderedSet[str]
    mutated_input_idxs: list[int]
    name_to_buffer: dict[str, ir.Buffer]
    name_to_users: defaultdict[str, list[ir.IRNode]]
    name_to_op: dict[str, ir.Operation]
    creation_time: Incomplete
    name: Incomplete
    cpp_wrapper: Incomplete
    record_multi_kernel_choice: Incomplete
    multi_kernel_to_choice: dict[str, str]
    aot_mode: Incomplete
    graph_id: Incomplete
    post_grad_graph_id: Incomplete
    scheduler: torch._inductor.scheduler.Scheduler
    autotuning_inputs: list[torch.Tensor] | None
    autotuning_mapping: dict[str, dict[str, int]] | None
    autotuning_grids: dict[str, Any] | None
    current_device: torch.device | None
    nodes_prefer_channels_last: Incomplete
    _warned_fallback: Incomplete
    user_visible_output_strides: Incomplete
    cache_key: str
    cache_path: str
    cache_linemap: list[tuple[int, str]]
    disable_cudagraphs_reason: str | None
    device_node_mapping: dict[torch.device, torch.fx.Node]
    orig_gm: torch.fx.GraphModule
    dynamo_flat_name_to_original_fqn: Incomplete
    allocated_constant_name: dict[str, str]
    get_backend_features: Incomplete
    effectful_ops: dict[_EffectType, ir.Buffer]
    unaligned_buffers: OrderedSet[str]
    no_fuse_buffer_names: OrderedSet[str]
    low_precision_codegen_ops: OrderedSet[str]
    invoke_quant_ops: OrderedSet[str]
    all_codegen_kernel_names: OrderedSet[str]
    workspace_id: Incomplete
    placeholder_idx: int
    bw_donated_idxs: Incomplete
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: Sequence[object] | None = None, shape_env: ShapeEnv | None = None, graph_id: int | None = None, cpp_wrapper: bool = False, aot_mode: bool = False, layout_opt: bool | None = None, extern_node_serializer: Callable[[list[ir.ExternKernelNode]], Any] | None = None, is_inference: bool = False, is_backward: bool = False, is_const_graph: bool = False, const_output_index: dict[str, int] | None = None, const_wrapper_code: str | None = None, const_kernel_code: str | None = None, const_module: GraphLowering | None = None, name: str | None = None, inputs_to_check: Sequence[int] | None = None) -> None: ...
    def freeze_runtime_asserts(self) -> None: ...
    def symbolic_sizes_strides(self, ex: torch.Tensor) -> tuple[Sequence[int | Expr], Sequence[int | Expr]]:
        """
        Support dynamic shapes and dynamic strides by assigning variables
        to each dimension.  We duck-shape tensors, so if two tensors
        have the same size they get assigned the same symbolic variable.
        """
    def static_sizes_strides(self, ex: torch.Tensor) -> tuple[list[sympy.Expr], list[sympy.Expr]]:
        """
        Primarily used to weights
        """
    def get_allocation_size(self, node: ir.TensorBox | ir.StorageBox | ir.Buffer | WorkspaceArg | ir.TorchBindObject) -> Sequence[Expr]: ...
    def get_allocation_storage_size(self, node: ir.Buffer | WorkspaceArg | ir.TorchBindObject) -> Expr: ...
    def has_feature(self, device: torch._inductor.ir.IRNode | device | None, feature: BackendFeature) -> bool: ...
    def get_current_device_or_throw(self) -> torch.device: ...
    @contextlib.contextmanager
    def set_current_device(self, device: torch.device) -> Iterator[None]: ...
    def get_training_phase(self) -> str: ...
    @staticmethod
    def decide_layout_opt(gm: GraphModule, *, is_inference: bool) -> bool:
        """
        Decide if we should enable layout optimization for this graph based on
        heuristics.
        """
    def qualify_name(self, name: str) -> str:
        """Prepend the given name with the graph name if any."""
    def make_subgraph(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor], subgraph_name: str) -> SubgraphLowering:
        """
        Make a subgraph of the current graph with all inherited parts, except
        the graph module (`gm`) and `example_inputs`.  The subgraphs are lowered
        separately and lifted into a separate function in the parent output
        wrapper code.  The subgraph name is qualified by the parent graph's
        name. Note that the lifting of subgraph is supported for python wrapper
        only. For cpp wrapper, we inline the subgraphs in the parent wrapper.
        """
    def find_nodes_prefer_channels_last(self) -> OrderedSet[Node]:
        """
        The rule to decide if an node prefer channels last is simple.
        1. if it's input/output of a convolution
        2. if one of its user prefers channels last

        We have rule 1 because cudnn runs a faster convolution kernel for channels last inputs;
        Rule 2 is also important. It makes sure that indirect inputs to convolution also prefers
        channels last.

        Consider the scenario: conv -> batch-norm -> relu -> conv
        Without rule 2, batch-norm output may use a contiguous layout. That will cause 2 extra copies:
        1. the output of batch-norm should be channels last initially since its input is a conv's output.
           Forcing the batch-norm's output to be contiguous results in the first copy
        2. The second conv's input is initially contiguous. This layout is propagated from the batch-norm's output.
           We need convert it to channels last layout which results in the second copy.
        With rule 2, we makes sure all the tensors in the chain uses channels last layout. So both copies
        can be saved.
        """
    def warn_fallback(self, name: str) -> None: ...
    def add_device_info(self, device: torch.device) -> None: ...
    @property
    def fake_mode(self) -> torch._subclasses.fake_tensor.FakeTensorMode: ...
    def try_get_buffer(self, buffer_name: str) -> ir.TensorBox | ir.Buffer | ir.TorchBindObject | None: ...
    def add_symbol_graph_input(self, symbol: sympy.Expr) -> None: ...
    def get_buffer(self, buffer_name: str) -> ir.TensorBox | ir.Buffer | ir.TorchBindObject: ...
    def get_dtype(self, buffer_name: str) -> torch.dtype: ...
    def get_numel(self, buffer_name: str) -> int | Expr: ...
    def run(self, *args: Any) -> Any: ...
    def register_operation(self, op: ir.Operation) -> str: ...
    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False) -> str: ...
    def register_operation_list(self, operation_names: list[str]) -> str: ...
    def register_users_of(self, node_output: Iterable[ir.IRNode] | ir.IRNode) -> None: ...
    def mark_buffer_mutated(self, name: str) -> None:
        """
        When a buffer is mutated we need to make sure all the reads to
        the old version are realized before the mutation happens.
        """
    def get_original_value_of_constant(self, name: str) -> torch.Tensor:
        """
        In AOTI, module buffers may have been mutated during the tracing and compilation.
        Thus we need to read from previously stored original buffers, to make sure the
        generated model.so uses correct initial values.
        """
    def allocate_non_dup_const_name(self, name: str | None, data: Tensor) -> str: ...
    def add_tensor_constant(self, data: Tensor, name: str | None = None) -> TensorBox: ...
    def constant_name(self, name: str, device_override: torch.device | None) -> str:
        """
        We AOT copy constants to the devices they are needed on.
        If device_override doesn't match the constant's device, then
        copy it and return a different name.
        """
    def placeholder(self, target: str, args: tuple[object], kwargs: dict[str, object]) -> Expr | TensorBox | None: ...
    def call_function(self, target: Callable, args: Any, kwargs: dict[str, Any]) -> Any: ...
    @staticmethod
    def can_inline_constant(t: torch.Tensor) -> bool:
        """
        True if this is a small constant attr that will be inlined.
        """
    def get_attr(self, target: str, args: tuple[()], kwargs: dict[str, object]) -> Constant | TensorBox | ir.Subgraph | TorchBindObject: ...
    def call_module(self, target: Any, args: Any, kwargs: Any) -> NoReturn: ...
    def call_method(self, target: Any, args: Any, kwargs: Any) -> NoReturn: ...
    def output(self, target: str, args: tuple[object], kwargs: dict[str, object]) -> None: ...
    def finalize(self) -> None: ...
    @contextmanager
    def set_current_node(self, node: torch.fx.Node): ...
    @contextmanager
    def set_current_wrapper_code(self) -> Iterator[None]: ...
    def propagate_mutation(self, fx_node: torch.fx.Node, old_args: tuple[Any], old_kwargs: dict[str, Any], new_args: tuple[Any], new_kwargs: dict[str, Any]) -> None:
        """Propagate mutations on new_args/new_kwargs back to old_args/old_kwargs.

        Assumes we may have cloned old_args/old_kwargs into new_args/new_kwargs
        and then called fx_node(*new_args, **new_kwargs).

        If fx_node mutates any of new_args/new_kwargs, and they are different from
        old_args/old_kwargs, then we need to update the original tensor.
        """
    def run_node(self, n: torch.fx.Node) -> object: ...
    def create_deferred_runtime_asserts(self, n: torch.fx.Node, new_unbacked_defs: OrderedSet[sympy.Symbol]) -> None: ...
    def validate_can_generate_cpp_wrapper(self) -> None: ...
    def init_wrapper_code(self, is_subgraph: bool = False, subgraph_name: str | None = None, parent_wrapper_code: PythonWrapperCodegen | None = None, partition_signatures: GraphPartitionSignature | None = None) -> None: ...
    def extract_autotune_inputs(self, example_inputs: list[int | float | torch.Tensor]) -> None: ...
    def codegen_with_cpp_wrapper(self) -> tuple[ValueWithLineMap, ValueWithLineMap]:
        """
        For GPU, Triton kernels are autotuned and stored as cubin files
        """
    def _update_scheduler(self) -> None:
        """
        (Re)initializes the scheduler member.  When initializing the scheduler, no CUBIN
        files should be generated (to avoid biasing any benchmarks and pessimizing
        fusion decisions).
        """
    def codegen(self) -> tuple[ValueWithLineMap, ValueWithLineMap]: ...
    def codegen_subgraph(self, parent_graph: GraphLowering) -> None:
        """
        This is a more compact version of the `codegen()` above
        where we codegen this graph as a subgraph of some parent
        graph. The parent graph is passed as an argument: the
        intention is to inline codegening of the subgraph in
        the parent graph's wrapper code (including the generated
        kernels). The wrapper code is not finalized (via `.generate()`
        call), as this will be done in the parent graph's `codegen()`.
        """
    def count_bytes(self) -> tuple[int, list[tuple[BaseSchedulerNode, int]], list[tuple[BaseSchedulerNode, float]]]: ...
    save_output_code: Callable[[str], None] | None
    def compile_to_module(self) -> CompiledModule: ...
    def _compile_to_module(self) -> CompiledModule: ...
    def _compile_to_module_lines(self, wrapper_code: ValueWithLineMap) -> CompiledModule: ...
    def get_output_names(self) -> list[str]: ...
    def is_unspec_arg(self, name: str) -> bool: ...

class SubgraphLowering(GraphLowering):
    """
    Mostly a helper class for the subgraph lowering. The main goal is to call
    init_wrapper_code with the subgraph related arguments.
    """
    parent: Incomplete
    def __init__(self, parent: GraphLowering, *args: Any, **kwargs: Any) -> None: ...
    def init_wrapper_code(self, is_subgraph: bool = False, subgraph_name: str | None = None, parent_wrapper_code: PythonWrapperCodegen | None = None, partition_signatures: GraphPartitionSignature | None = None) -> None: ...
