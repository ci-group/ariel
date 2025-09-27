import dataclasses
import torch
from .. import config as config
from .autograd_cache import AOTAutogradCache as AOTAutogradCache, serialize_graph_module as serialize_graph_module, should_use_remote_autograd_cache as should_use_remote_autograd_cache
from .dispatch_and_compile_graph import aot_dispatch_autograd_graph as aot_dispatch_autograd_graph, aot_dispatch_base_graph as aot_dispatch_base_graph
from .logging_utils import track_graph_compiling as track_graph_compiling
from .runtime_wrappers import AOTDedupeWrapper as AOTDedupeWrapper, AOTDispatchAutograd as AOTDispatchAutograd, AOTDispatchSubclassWrapper as AOTDispatchSubclassWrapper, AOTSyntheticBaseWrapper as AOTSyntheticBaseWrapper, AutogradLazyBackwardCompileInfo as AutogradLazyBackwardCompileInfo, CompilerWrapper as CompilerWrapper, DebugAssertWrapper as DebugAssertWrapper, EffectTokensWrapper as EffectTokensWrapper, FakifiedOutWrapper as FakifiedOutWrapper, FunctionalizedRngRuntimeWrapper as FunctionalizedRngRuntimeWrapper, RuntimeWrapper as RuntimeWrapper, make_runtime_safe as make_runtime_safe, post_compile as post_compile, pre_compile as pre_compile
from .schemas import AOTConfig as AOTConfig, MutationType as MutationType, ViewAndMutationMeta as ViewAndMutationMeta
from .subclass_utils import compute_inner_mutated_inp_indices_from_subclass_meta as compute_inner_mutated_inp_indices_from_subclass_meta
from .utils import _get_symint_hints as _get_symint_hints, contain_metadata_mutation_ops as contain_metadata_mutation_ops, get_cuda_generator_meta_val as get_cuda_generator_meta_val, make_boxed_func as make_boxed_func, strict_zip as strict_zip, unlift_tokens as unlift_tokens
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._dynamo.utils import detect_fake_mode as detect_fake_mode, dynamo_timed as dynamo_timed, lazy_format_graph_code as lazy_format_graph_code
from torch._guards import CompileContext as CompileContext, TracingContext as TracingContext
from torch._logging import getArtifactLogger as getArtifactLogger, trace_structured as trace_structured
from torch._subclasses import FakeTensor as FakeTensor
from torch._subclasses.meta_utils import is_sparse_any as is_sparse_any
from torch.fx.experimental._backward_state import BackwardState as BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node as is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals as fx_placeholder_vals
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.passes._tensorify_python_scalars import tensorify_python_scalars as tensorify_python_scalars
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.types import py_sym_types as py_sym_types
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass
from typing import Any, Callable

zip = strict_zip
log: Incomplete
aot_joint_log: Incomplete
aot_graphs_log: Incomplete
aten: Incomplete
DispatchReturn = tuple[Callable, ViewAndMutationMeta]

def _create_wrappers_for_dispatch(needs_autograd: bool) -> list[CompilerWrapper]:
    """
    Wrappers that run on every dispatch function
    """
def aot_dispatch_export(flat_fn: Callable, flat_args: list[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta, needs_autograd: bool) -> DispatchReturn: ...
def sanitize_aot_config(input: AOTConfig) -> AOTConfig: ...
def aot_dispatch_base(flat_fn, flat_args: list[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> DispatchReturn:
    """
    Handles functions that don't need autograd. Runs wrappers and compiles with fw_compiler.
    """
def collect_fw_donated_buffer_idxs(fw_ins: list[FakeTensor | None], user_fw_outs: list[FakeTensor | None], bw_outs: list[FakeTensor | None], saved_tensors: list[FakeTensor]) -> list[int]:
    """
    Checks if the saved tensors are donated buffers, which means a saved tensor is not
    an alias of any tensors in fw_ins, user_fw_outs, and bw_outs.
    """
def collect_bw_donated_buffer_idxs(fw_module: torch.fx.GraphModule, bw_module: torch.fx.GraphModule, fw_metadata: ViewAndMutationMeta) -> list[int]:
    """
    Collects backward donated buffer indexes from fw_module and bw_module.
    """

@dataclasses.dataclass
class InvokeSubgraphHopGraphs:
    """
    A data structure to hold all the information needed to partition the
    `joint_hop_gm` and joint graph and the restitch the `new_fw_hop_gm` and
    `new_bw_hop_gm` into the bigger `joint_gm`.
    """
    partitioning_done: bool = ...
    old_num_fw_outputs: int | None = ...
    old_num_fw_inputs: int | None = ...
    new_fw_hop_gm: torch.fx.GraphModule | None = ...
    new_bw_hop_gm: torch.fx.GraphModule | None = ...
    new_num_sym_nodes: int | None = ...
    new_num_saved_nodes: int | None = ...

def run_joint_graph_passes_on_hops(joint_gm: torch.fx.GraphModule, joint_inputs: Any, aot_config: AOTConfig) -> torch.fx.GraphModule:
    """
    This pass runs the joint graph passes on the HOP graph. In torch.compile, we
    typically have many passes which work on the joint graph and then end with a
    partitioner.


    The partitioner part is quite mechanical to handle. HOP have their own
    forward and backward graph. The process can be broken into following steps

    1) Get a `joint_hop_gm` from the `fw_hop_gm` and `bw_hop_gm`
    2) Run joint graph passes on the `joint_hop_gm` to get `new_fw_hop_gm` and `new_bw_hop_gm`
    3) Stitch the `new_fw_hop_gm` and `new_bw_hop_gm` back into the `joint_gm`.

    The terminology used in the code is
    `joint_graph/joint_gm` : Refers to the main graph. This may contain many HOPs which have their own `hop_graph`
    `fw_hop_graph/fw_hop_gm` : Refers to the forward graph associated with a HOP.
    `bw_hop_graph/bw_hop_gm` : Refers to the backward graph associated with a HOP.
    `joint_hop_graph/joint_hop_gm` : Refers to the subgraph associated with the HOP like invoke_subgraph.
    `new_fw_hop_graph/new_fw_hop_gm` : Refers to the forward graph after partitioning is applied to `joint_hop_gm`.
    `new_bw_hop_graph/new_bw_hop_gm` : Refers to the backward graph after partitioning is applied to `joint_hop_gm`.

    NB: This pass works for invoke_subgraph today because we took extra care in
    the Autograd.Dispatch key of invoke_subgraph to vastly simplify Step 1.
    """
def maybe_log_graph(gm, graph_name, aot_config, structured_log_prefix_fn, out_structured_logs: list[str] | None = None): ...
def create_wrap_fn(fn, args): ...
def prepare_hook_gm(aot_config, fn, args): ...
def maybe_inline_graph_saved_tensors_hooks(fw_module, bw_module, num_inner_fwd_outputs, inner_meta, aot_config, static_input_indices): ...
def aot_dispatch_autograd(flat_fn, flat_args: list[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> DispatchReturn:
    """
    Autograd logic. Generates a joint graph, partitions it, manipulates the input with various wrappers,
    and returns a wrapped torch.autograd.Function with a forward and backward.
    """
