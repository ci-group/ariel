import torch
from .. import config as config, ir as ir, pattern_matcher as pattern_matcher
from ..codegen.common import custom_backend_passes as custom_backend_passes
from ..comms import remove_fsdp2_unsharded_param_graph_input_usage as remove_fsdp2_unsharded_param_graph_input_usage
from ..fx_utils import FakeTensorUpdater as FakeTensorUpdater, get_fake_args_kwargs as get_fake_args_kwargs, get_node_storage as get_node_storage
from ..pattern_matcher import Arg as Arg, CallFunction as CallFunction, CallFunctionVarArgs as CallFunctionVarArgs, Ignored as Ignored, KeywordArg as KeywordArg, ListOf as ListOf, MULTIPLE as MULTIPLE, Match as Match, MultiOutputPattern as MultiOutputPattern, PatternMatcherPass as PatternMatcherPass, _return_true as _return_true, filter_nodes as filter_nodes, fwd_only as fwd_only, get_arg_value as get_arg_value, get_mutation_region_id as get_mutation_region_id, init_once_fakemode as init_once_fakemode, register_graph_pattern as register_graph_pattern, register_replacement as register_replacement, stable_topological_sort as stable_topological_sort
from ..utils import OPTIMUS_EXCLUDE_POST_GRAD as OPTIMUS_EXCLUDE_POST_GRAD, decode_device as decode_device, get_all_devices as get_all_devices, get_gpu_type as get_gpu_type, is_gpu as is_gpu, is_pointwise_use as is_pointwise_use
from ..virtualized import V as V
from .b2b_gemm import B2B_GEMM_PASS as B2B_GEMM_PASS
from .ddp_fusion import fuse_ddp_communication as fuse_ddp_communication
from .group_batch_fusion import POST_GRAD_FUSIONS as POST_GRAD_FUSIONS, group_batch_fusion_passes as group_batch_fusion_passes
from .micro_pipeline_tp import micro_pipeline_tp_pass as micro_pipeline_tp_pass
from .pre_grad import is_same_dict as is_same_dict, save_inductor_dict as save_inductor_dict
from .reinplace import reinplace_inplaceable_ops as reinplace_inplaceable_ops
from .split_cat import POST_GRAD_PATTERNS as POST_GRAD_PATTERNS
from _typeshed import Incomplete
from torch import fx as fx
from torch._decomp import register_decomposition as register_decomposition
from torch._dynamo.utils import counters as counters
from torch._inductor import comms as comms
from torch._inductor.virtualized import ops as ops
from torch._logging import trace_structured as trace_structured
from torch._prims_common import is_boolean_dtype as is_boolean_dtype, is_expandable_to as is_expandable_to, is_integer_dtype as is_integer_dtype
from torch.fx.experimental.symbolic_shapes import statically_known_true as statically_known_true, sym_eq as sym_eq
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

_T = TypeVar('_T')
_P = ParamSpec('_P')
log: Incomplete
aten: Incomplete
prims: Incomplete
pass_patterns: Incomplete

def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
def prepare_softmax_pattern(x, dim): ...
def prepare_softmax_replacement(x, dim):
    """
    Return xsub since otherwise log-softmax can not be matched
    due to a use of this intermediate node. Same reason to return
    xsub.exp() for softmax.
    """
def prepare_softmax_extra_check(match):
    """
    We only have triton online softmax kernels currently.
    """
def decompose_map_to_while_loop(gm: torch.fx.GraphModule):
    """This is similar to decompose_scan_to_while_loop."""
def resolve_shape_to_proxy(shape: list[int | torch.SymInt], bound_symbols: dict[Any, Any]):
    """
    Given a list of symints/ints, this function returns a calculated expression of bound_symbols' values.
    When we trace this function, we'll get a graph with call_function nodes that describes how the shape expr is
    computed from bound_symbols' values.

    Suppose shape = (s1*s2, s1+s2) and bound_symbols = {s1: arg0, s2: arg1}, the result will be
    (arg0 * arg1, arg0 + arg1).
    """
def decompose_scan_to_while_loop(gm: torch.fx.GraphModule):
    """
    NOTE [decompose scan to while_loop]
    This pass decomposes `scan` to  `while_loop` by replacing the scan fx_node with a while_loop hop.

    Suppose we have a function f:

        def f():
            init = torch.zeros([])
            xs = torch.arange(4)
            ys = []
            for i in range(xs.size(0)):
                init = xs[i] + init
                ys.append(init)

            # Return the final carry and stack the intermediates
            return init, torch.stack(ys)

    We could rewrite it with a scan with the benefits of reducing compilation time/binary size, reducing
    memory usage, supporting loops over unbacked shapes and cudagraph etc.

        def g():
            def step_fn(init: torch.Tensor, x: torch.Tensor):
                next_init = x + init
                return next_init, next_init

            init = torch.zeros([])
            xs = torch.arange(4)
            final_carry, ys = torch._higher_order.scan(step_fn, init, xs)
            return final_carry, ys

    This pass will rewrite scan into:

        def k():
            init = torch.zeros([])
            xs = torch.arange(4)

            # we create a loop_idx and loop through xs.shape[0]
            loop_idx = torch.zeros([])
            ys = torch.empty_strided(_shape_stride_of_ys)
            def cond_fn(loop_idx, ys, init, xs):
                return loop_idx < xs.shape[0]

            # we pre-allocate the output buffer ys and inplace
            # copy the y of each intermediate into a slice.
            # NOTE [Pre-allocate scan's output buffer].
            def body_fn(loop_idx, ys, init, xs):
                int_idx = loop_idx.item()
                next_init, y = step_fn(init, xs[int_idx])
                ys[int_idx].copy_(y)
                return loop_idx + 1, ys, next_init, xs

            final_carry, _, _, ys = torch._higher_order.while_loop(cond_fn, body_fn, (loop_idx, ys, init, xs))
            return final_carry, ys
    """
@init_once_fakemode
def lazy_init() -> None: ...
def reorder_for_locality(graph: torch.fx.Graph): ...
def register_lowering_pattern(pattern, extra_check=..., pass_number: int = 1) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register an aten to inductor IR replacement pattern
    """
def is_valid_mm_plus_mm(match: Match): ...
def scatter_upon_const_tensor_extra_check(m): ...
def scatter_upon_const_tensor(match: Match, shape, background_val, dtype, dim, selector, val):
    """
    Match the pattern of full+scatter into a pointwise.

    TODO: Right now the scatter value must be a scalar. But we could support it
    when it is a tensor as well.
    """
def mm_plus_mm(match: Match, mat1, mat2, mat3, mat4): ...
def pointless_cumsum_replacement(match: Match, shape, fill_value, device, dtype, dim):
    """Based on a pattern in OPTForCausalLM"""

_cat_1: Incomplete

def cat_slice_cat(match, cat_input, size, dim: int = 1):
    """
    This is an example of a more complex pattern where cat_1 is used
    multiple times inside the pattern.  We fold 2 calls to cat into one.

    Matches:
        cat_1: f32[1024, 4077] = torch.ops.aten.cat.default([add_26, primals_217], 1)
        slice_1: f32[1024, 4077] = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
        slice_2: f32[1024, 19] = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
        cat_2: f32[1024, 4096] = torch.ops.aten.cat.default([cat_1, slice_2], 1)


    Rewrite to:
        slice_2 = torch.ops.aten.slice.Tensor(add_26, 1, 0, 19)
        cat_2 = torch.ops.aten.cat.default([add_26, primals_217, slice2], 1)
    """
def is_valid_splitwithsizes_cat(match): ...
def same_meta(node1: torch.fx.Node, node2: torch.fx.Node):
    """True if two nodes have the same metadata"""

noop_registry: dict[Any, Any]

def register_noop_decomp(targets, nop_arg: int = 0): ...
def slice_noop(self, dim: int = 0, start=None, end=None, step: int = 1): ...
def slice_scatter_noop(self, src, dim: int = 0, start=None, end=None, step: int = 1): ...
def repeat_noop(self, repeats): ...
def constant_pad_nd(x, padding, fill_value: int = 0): ...
def convert_element_type_noop(x, dtype: torch.dtype): ...
def device_put_noop(x, device, non_blocking: bool = True): ...
def int_noop(x): ...
def pow_noop(a, b): ...
def cat_noop(inputs, dim: int = 0): ...
def view_default_noop(arg, size): ...
def view_dtype_noop(arg, dtype): ...
def true_noop(*args, **kwargs): ...
def remove_noop_ops(graph: torch.fx.Graph):
    """
    Removes both operations that are essentially aten.clone and operations that are essentially aten.alias from the graph.
    """
def remove_assert_ops(graph: torch.fx.Graph):
    """
    Removes aten._assert_tensor_metadata.default op because
    1) it will be lowered to a no-op in inductor
    2) it can block fusion, such as unfuse_bias_add_to_pointwise fusion.

    This op could come from aten.to functionalization in export.

    For example, if we have a graph like below

    %addmm = aten.addmm.default(%linear_bias, %arg3_1, %permute)
    %_assert_tensor_metadata = aten._assert_tensor_metadata.default(%addmm, None, None, torch.float16)
    %convert_element_type_3 = prims.convert_element_type.default(%addmm, torch.float32)
    %pow_1 = aten.pow.Tensor_Scalar(%convert_element_type_3, 2)

    We still want to fuse add from addmm with pow, instead of fusing add with mm, according to unfuse_bias_add_to_pointwise fusion.

    However, aten._assert_tensor_metadata.default is not a pointwise op, and would fail the should_prefer_unfused_addmm check.

    We remove this op so it doesn't block fusion decisions. It's safe because this op is lowered to a no-op with @register_lowering.

    """
def decompose_triton_kernel_wrapper_functional(graph):
    """Decomposes triton_kernel_wrapper_functional nodes into clones and the underlying
    mutation node.

    We assume that the reinplacing pass runs before this; the reinplacing pass
    tells us (via rewriting the arguments or .meta to those nodes) which
    Tensors we should clone and which Tensors are safe to reinplace.
    """
def decompose_auto_functionalized(graph):
    """Decomposes auto_functionalized nodes into clones and the underlying
    mutation node.

    We assume that the reinplacing pass runs before this; the reinplacing pass
    tells us (via rewriting the arguments or .meta to those nodes) which
    Tensors we should clone and which Tensors are safe to reinplace.
    """
def splitwithsizes_cat_replace(match, input_): ...
def is_valid_cat_splitwithsizes(match): ...
def cat_splitwithsizes_replace(match, input_): ...
def view_to_reshape(gm) -> None:
    """
    Replace view ops in the GraphModule to reshape ops.
    """
def should_prefer_unfused_addmm(match): ...
def unfuse_bias_add_to_pointwise(match: Match, mat1, mat2, *, inp): ...
def is_valid_addmm_fusion(match): ...
def addmm(match, mat1, mat2, *, inp): ...
def register_partial_reduction_pattern():
    """Reuse partial reductions in complete reductions"""
def check_shape_cuda_and_fused_int_mm_mul_enabled(match): ...
def is_index_put_and_requires_h2d_sync_for_gpu_value(node): ...

class ConstructorMoverPass:
    target: Incomplete
    allow_inputs: Incomplete
    allow_outputs: Incomplete
    def __init__(self, target: str, allow_outputs: bool = False, allow_inputs: bool = False) -> None:
        """
        Move constructors from cpu to the target_device.

        Sweeps through the module, looking for constructor nodes that can be moved
        to the target_device.

        A constructor node can be moved to the target_device iff all of its users
        can also be moved (tested by cannot_be_moved). Otherwise, all dependent
        constructor nodes won't be moved.

        - target: target device type
        - allow_outputs: allow outputs to be moved
        - allow_inputs: allow inputs to be moved
        """
    def allow_cpu_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node that returns a tensor on the target device may have
        cpu tensors as input.
        """
    def is_on_target_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node is on the target device.
        """
    def is_cpu_scalar_tensor(self, node: fx.Node) -> bool:
        """
        Returns whether a node is a cpu scalar tensor.
        """
    def all_inputs_are_cpu_scalar_or_on_target_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node's inputs are either cpu scalar tensors or
        on the target device.
        """
    def cannot_be_moved(self, node: fx.Node) -> bool:
        """
        Returns whether a node can be moved to the target device.

        If this function returns False, it means that this node and all of its users
        won't be moved into the target device.
        """
    def get_node_device(self, node: fx.Node) -> torch.device | None:
        """
        Get the device of a node.
        """
    def get_cpu_indeg_count(self, graph: fx.Graph) -> dict[fx.Node, int]:
        """
        Get the number of cpu inputs to a node
        """
    def __call__(self, graph: fx.Graph) -> None: ...
    def find_movable_constructors(self, graph: fx.Graph, constructors: list[fx.Node]) -> OrderedSet[fx.Node]:
        """
        Starting from the cpu constructors, iterate through the graph and test that all of their
        downstream uses can safely be moved to cpu.
        """

def move_constructors_to_gpu(graph: fx.Graph) -> None:
    """
    Moves intermediary tensors which are constructed on the cpu to gpu when safe
    """
