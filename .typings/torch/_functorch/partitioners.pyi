import functools
import logging
import torch
import torch.fx as fx
from . import config as config
from ._activation_checkpointing.graph_info_provider import GraphInfoProvider as GraphInfoProvider
from ._activation_checkpointing.knapsack import dp_knapsack as dp_knapsack, greedy_knapsack as greedy_knapsack, ilp_knapsack as ilp_knapsack
from ._activation_checkpointing.knapsack_evaluator import KnapsackEvaluator as KnapsackEvaluator
from ._aot_autograd.logging_utils import get_aot_graph_name as get_aot_graph_name
from ._aot_autograd.utils import get_cuda_generator_meta_val as get_cuda_generator_meta_val, is_with_effects as is_with_effects
from .compile_utils import fx_graph_cse as fx_graph_cse, get_aten_target as get_aten_target, raise_getitems as raise_getitems
from _typeshed import Incomplete
from dataclasses import dataclass
from torch._dynamo.utils import counters as counters, is_node_meta_valid as is_node_meta_valid
from torch._functorch._activation_checkpointing.ac_logging_utils import create_structured_trace_for_min_cut_info as create_structured_trace_for_min_cut_info
from torch._logging import trace_structured as trace_structured
from torch._subclasses.fake_tensor import extract_tensor_metadata as extract_tensor_metadata
from torch.fx.experimental._backward_state import BackwardState as BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node as is_sym_node, py_sym_types as py_sym_types
from torch.fx.experimental.sym_node import magic_methods as magic_methods, method_to_operator as method_to_operator
from torch.fx.experimental.symbolic_shapes import find_symbol_binding_fx_nodes as find_symbol_binding_fx_nodes, free_symbols as free_symbols, hint_int as hint_int, is_symbol_binding_fx_node as is_symbol_binding_fx_node, statically_known_false as statically_known_false, statically_known_true as statically_known_true
from torch.fx.passes import graph_drawer as graph_drawer
from torch.utils._mode_utils import no_dispatch as no_dispatch
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils.checkpoint import CheckpointPolicy as CheckpointPolicy
from typing import Callable

AOT_PARTITIONER_DEBUG: bool
log: logging.Logger
aten: Incomplete
prims: Incomplete

@dataclass
class OpTypes:
    """Class for keeping track of different operator categories"""
    fusible_ops: OrderedSet[Callable]
    compute_intensive_ops: OrderedSet[Callable]
    random_ops: OrderedSet[Callable]
    view_ops: OrderedSet[Callable]
    recomputable_ops: OrderedSet[Callable]
    def is_fusible(self, node: fx.Node): ...
    def is_compute_intensive(self, node: fx.Node): ...
    def is_random(self, node: fx.Node): ...
    def is_view(self, node: fx.Node): ...
    def is_recomputable(self, node: fx.Node): ...

@dataclass
class NodeInfo:
    inputs: list[fx.Node]
    _required_fw_nodes: OrderedSet[fx.Node]
    required_bw_nodes: OrderedSet[fx.Node]
    unclaimed_nodes: OrderedSet[fx.Node]
    fw_order: dict[fx.Node, int]
    static_lifetime_input_nodes: OrderedSet[fx.Node]
    @functools.cached_property
    def required_fw_nodes(self) -> list[fx.Node]: ...
    def is_required_fw(self, n: fx.Node) -> bool: ...
    def is_required_bw(self, n: fx.Node) -> bool: ...
    def is_unclaimed(self, n: fx.Node) -> bool: ...
    def get_fw_order(self, n: fx.Node) -> int: ...

@dataclass
class MinCutOptions:
    ban_if_used_far_apart: bool
    ban_if_long_fusible_chains: bool
    ban_if_materialized_backward: bool
    ban_if_not_in_allowlist: bool
    ban_if_reduction: bool

def must_recompute(node: fx.Node) -> bool: ...
def has_recomputable_ops(fx_g: fx.GraphModule) -> bool: ...
def has_recomputable_rng_ops(fx_g: fx.GraphModule) -> bool: ...
def sym_node_size(node: fx.Node) -> int: ...

class InvalidNodeBase:
    def __repr__(self) -> str: ...

InvalidNode: Incomplete

def _extract_graph_with_inputs_outputs(joint_graph: fx.Graph, inputs: list[fx.Node], outputs: list[fx.Node], subgraph: str | None = None) -> fx.Graph:
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as
    inputs and returns the specified outputs.

    This includes specifying non-placeholder nodes as inputs.

    The general strategy is to initialize all inputs with proxies as we
    encounter them, and trace through the graph, only keeping values which take
    in valid proxies. Then, all dead code is eliminated.
    """
def _is_primal(node: fx.Node) -> bool: ...
def _is_tangent(node: fx.Node) -> bool: ...
def _is_bwd_seed_offset(node: fx.Node) -> bool: ...
def _is_fwd_seed_offset(node: fx.Node) -> bool: ...
def _is_backward_state(node: fx.Node) -> bool: ...
def _has_tag_is_backward(node: fx.Node) -> bool: ...
def _has_tag_must_be_in_backward(node: fx.Node) -> bool: ...
def _must_be_in_backward(node: fx.Node) -> bool: ...
def _extract_fwd_bwd_outputs(joint_module: fx.GraphModule, *, num_fwd_outputs) -> tuple[list[fx.Node], list[fx.Node]]: ...
def _remove_by_name(saved_values: list[fx.Node], name: str): ...
def find_first_sym_node(fwd_module_outputs: list[fx.Node] | tuple[fx.Node]) -> int: ...
def calculate_quantization_scaling(graph: torch.fx.Graph, node: torch.fx.Node, max: float = 57344.0, min: float = 1e-12): ...
def perform_quantization(graph: torch.fx.Graph, node: torch.fx.Node, scale_node: torch.fx.Node, quant_type: torch.dtype, clamp_min: float, clamp_max: float) -> torch.fx.Node: ...
def calculate_tensor_size(tensor: torch.Tensor) -> float:
    """
    Calculate the size of a PyTorch tensor in megabytes (MB).

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        float: Memory size in MB
    """
def get_allowed_dtypes() -> list[torch.dtype]: ...
def should_quantize(node: torch.fx.Node) -> bool: ...
def get_quant_type() -> torch.dtype: ...
def calculate_range(dtype: torch.dtype) -> tuple:
    """
    Calculate the range of values for a given torch.dtype.
    Args:
        dtype (torch.dtype): The input dtype.
    Returns:
        tuple: A tuple containing the minimum and maximum values.
    """
def quantize_activation_fw(graph: torch.fx.Graph) -> None: ...
def quantize_activation_bw(graph: torch.fx.Graph) -> None: ...
def enable_activation_quantization(saved_values: list[fx.Node], fwd_module: fx.GraphModule, bwd_module: fx.GraphModule, static_lifetime_input_nodes: OrderedSet[fx.Node] | None = None) -> None: ...
def _extract_fwd_bwd_modules(joint_module: fx.GraphModule, saved_values: list[fx.Node], saved_sym_nodes: list[fx.Node], *, num_fwd_outputs: int, static_lifetime_input_nodes: OrderedSet[fx.Node] | None = None) -> tuple[fx.GraphModule, fx.GraphModule]: ...
def default_partition(joint_module: fx.GraphModule, _joint_inputs, *, num_fwd_outputs, static_lifetime_input_indices: list[int] | None = None, static_lifetime_input_nodes: OrderedSet[fx.Node] | None = None) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the :attr:`joint_module` in a manner that closely resembles the
    behavior observed in the original ``.forward()`` and ``.backward()`` of the
    callable, i.e., the resulting forward graph contains those operators that
    are executed in the original ``.forward()`` callable passed to
    :func:`aot_function`.

    The default partitioner collects the operators that are between the forward
    inputs and the forward outputs. This helps in finding the tensors which have
    to be stashed for the backward pass. These stashed tensors become the output
    of the generated forward graph. The remaining operators are then placed in
    the backward graph.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """

INT_INF: Incomplete

def _tensor_nbytes(numel: int, dtype) -> int: ...
def _size_of(node: fx.Node) -> int: ...
def _count_ops(graph: fx.Graph): ...
@functools.cache
def pointwise_ops(): ...
def sort_depths(args, depth_map: dict[fx.Node, int]) -> list[tuple[fx.Node, int]]: ...
def reordering_to_mimic_autograd_engine(gm: fx.GraphModule) -> fx.GraphModule:
    """
    This pass finds the first bwd node in the graph (by looking at users of
    tangents) and then reorders the graph by walking from this node to all the
    way to the end of the graph. At each op in this traveral, we insert this op
    in a new graph and try to bring only the relevant subgraph from the other
    non-bwd edges relevant for this op. This closely mimics the behavior of
    autograd engine.

    Why is this pass required in the first place?

    This is an artifact of how partitioners work today. The starting point of
    partitioner is a joint graph, which is fwd and then bwd graph. In the case
    of checkpointing, we keep portions of fwd graph in their original place in
    the joint graph, while obtaining a bwd graph. As a result, the resulting bwd
    graph has copies of recomputed fwd subgraphs followed by the original bwd
    graph. If we run this naively, this leads to bad memory footprint, because
    the fwd subgraphs are live for way longer duration than necessary. This pass
    reorders the operations such that we prioritize the ops for the original bwd
    graph while only realizing those ops from the fwd graph that are necessary
    at any given point in the graph.
    """
def apply_graphsafe_rng_functionalization(fw_module: torch.fx.GraphModule, bw_module: torch.fx.GraphModule, fw_node: torch.fx.Node, bw_node: torch.fx.Node, device: torch.device, rng_count: int, last_fwd_input: torch.fx.Node, last_bwd_input: torch.fx.Node):
    """
    Note [CUDA Graph Safe RNG Functionalization]

    CUDA Graph capture doesn't work with get_rng_state and set_rng_state because these functions operate on CPU values,
    while CUDA Graph RNG capture uses on-device CUDA tensors. To solve this, we use graphsafe_set_state with a
    CUDA Generator registered to the CUDA Graph before capture begins. graphsafe_set_state updates the generator's pointer
    to reference a different GeneratorImpl, ensuring subsequent calls are correctly forwarded to the desired generator
    (and its cuda-tensor RNG state during graph capture).

    For each RNG operation's forward/backward pair:

    - We create two generators initialized with identical values
    - Each forward and backward call advances its respective generator equally
    - This keeps generators synchronized so forward and backward operations use matching RNG values

    When forward is called multiple times before backward (causing desynchronization):

    - We save the forward RNG state
    - We update the backward Generator's state before executing backward

    Before each CUDA Graph replay, replay_prologue updates captured RNG pointers with current states, ensuring backward Generator
    changes are reflected during replay.

    This function modifies both forward and backward computation graphs by:

    Creating RNG state placeholders for both passes
    Updating the forward node to use graph-safe RNG state
    Updating the backward node to use graph-safe RNG state

    For more details: https://github.com/pytorch/pytorch/issues/113541
    """
def functionalize_rng_ops(joint_module: fx.GraphModule, fw_module: fx.GraphModule, bw_module: fx.GraphModule, num_sym_nodes: int) -> tuple[fx.GraphModule, fx.GraphModule]: ...
def force_save_collectives(joint_module: fx.GraphModule) -> None:
    """
    By default, the partitioner is not allowed to recompute collectives
    unless they come from a user-annotated AC region.
    See Note [Recomputing collectives in the partitioner]
    """
def cleanup_recompute_tags(joint_module: fx.GraphModule) -> fx.GraphModule:
    """
    If there are two consecutive checkpointed blocks with no operator in
    between, we would still want to stash the tensor at the boundary of
    checkpointed blocks. The following pass makes the last output node
    non-recomputable to allow for that.
    """
def solve_min_cut(joint_graph: fx.Graph, node_info: NodeInfo, min_cut_options: MinCutOptions, dont_ban: OrderedSet[fx.Node] | None = None): ...
def visualize_min_cut_graph(nx_graph) -> None: ...
def get_default_op_list() -> OpTypes: ...
def get_name_to_node(graph: fx.Graph): ...
def _optimize_runtime_with_given_memory(joint_graph: fx.Graph, memory: list[float], runtimes: list[float], max_memory: float, node_info: NodeInfo, all_recomputable_banned_nodes: list[fx.Node]) -> tuple[float, list[int], list[int]]: ...
def _remove_symbols_without_guarding(x: torch.Tensor, fallback: int) -> torch.Tensor: ...
def estimate_runtime(node): ...
def choose_saved_values_set(joint_graph: fx.Graph, node_info: NodeInfo, memory_budget: int = 1) -> list[fx.Node]: ...
def _broadcast_rank0_decision(joint_graph: torch.fx.Graph, saved_values: list[torch.fx.Node]): ...
def min_cut_rematerialization_partition(joint_module: fx.GraphModule, _joint_inputs, compiler: str = 'inductor', *, num_fwd_outputs, static_lifetime_input_indices: list[int] | None = None) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimination.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.
        _joint_inputs: The inputs to the joint graph. This is unused.
        compiler: This option determines the default set of recomputable ops.
            Currently, there are two options: ``nvfuser`` and ``inductor``.
        recomputable_ops: This is an optional set of recomputable ops. If this
            is not None, then this set of ops will be used instead of the
            default set of ops.
        num_fwd_outputs: The number of outputs from the forward graph.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """
def draw_graph(traced: torch.fx.GraphModule, fname: str, figname: str = 'fx_graph', clear_meta: bool = True, prog: str | list[str] | None = None, parse_stack_trace: bool = False, dot_graph_shape: str | None = None) -> None: ...
