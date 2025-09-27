import abc
import contextlib
import enum
import functools
import torch
import torch.fx
from . import config as config, metrics as metrics
from .._dynamo.backends.common import aot_autograd as aot_autograd
from .._dynamo.exc import ShortenTraceback as ShortenTraceback, SkipFrame as SkipFrame
from ..fx._lazy_graph_module import _use_lazy_graph_module as _use_lazy_graph_module
from ..fx.graph import _PyTreeCodeGen as _PyTreeCodeGen
from ..utils._triton import has_triton as has_triton
from .codegen.common import get_wrapper_codegen_for_device as get_wrapper_codegen_for_device, init_backend_registration as init_backend_registration
from .debug import DebugContext as DebugContext
from .decomposition import select_decomp_table as select_decomp_table
from .exc import InductorError as InductorError
from .fx_passes.joint_graph import joint_graph_passes as joint_graph_passes
from .fx_passes.post_grad import post_grad_passes as post_grad_passes, view_to_reshape as view_to_reshape
from .fx_passes.pre_grad import pre_grad_passes as pre_grad_passes
from .graph import GraphLowering as GraphLowering
from .ir import ExternKernelNode as ExternKernelNode, IRNode as IRNode, get_device_type as get_device_type
from .triton_bundler import TritonBundler as TritonBundler
from .utils import align_inputs_from_check_idxs as align_inputs_from_check_idxs, clone_preserve_strides as clone_preserve_strides, copy_misaligned_inputs as copy_misaligned_inputs, get_cloned_parameter_buffer_name as get_cloned_parameter_buffer_name, get_first_incompatible_cudagraph_node as get_first_incompatible_cudagraph_node, maybe_get_suppress_shape_guards_ctx as maybe_get_suppress_shape_guards_ctx, output_node as output_node, remove_unaligned_input_idxs as remove_unaligned_input_idxs, shape_env_from_inputs as shape_env_from_inputs
from .virtualized import V as V
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import AbstractContextManager
from torch import fx as fx
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._dynamo import compiled_autograd as compiled_autograd
from torch._dynamo.device_interface import get_interface_for_device as get_interface_for_device
from torch._dynamo.repro.after_aot import wrap_compiler_debug as wrap_compiler_debug
from torch._dynamo.utils import CompileEventLogger as CompileEventLogger, chromium_event_timed as chromium_event_timed, counters as counters, detect_fake_mode as detect_fake_mode, dynamo_timed as dynamo_timed, flatten_graph_inputs as flatten_graph_inputs, get_metrics_context as get_metrics_context, lazy_format_graph_code as lazy_format_graph_code, set_feature_use as set_feature_use
from torch._functorch._aot_autograd.schemas import FQN as FQN, GraphInputName as GraphInputName, GraphSignature as GraphSignature
from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters as unwrap_tensor_subclass_parameters
from torch._functorch.aot_autograd import SerializableAOTDispatchCompiler as SerializableAOTDispatchCompiler, aot_export_module as aot_export_module, make_boxed_func as make_boxed_func
from torch._inductor.codecache import FxGraphCache as FxGraphCache, code_hash as code_hash, output_code_log as output_code_log
from torch._inductor.cudagraph_utils import BoxedDeviceIndex as BoxedDeviceIndex, PlaceholderInfo as PlaceholderInfo, format_default_skip_message as format_default_skip_message, log_cudagraph_skip_and_bump_counter as log_cudagraph_skip_and_bump_counter
from torch._inductor.debug import save_args_for_compile_fx_inner as save_args_for_compile_fx_inner
from torch._inductor.output_code import CompiledAOTI as CompiledAOTI, CompiledFxGraph as CompiledFxGraph, CompiledFxGraphConstantsWithGm as CompiledFxGraphConstantsWithGm, OutputCode as OutputCode, _StrideExprStr as _StrideExprStr, get_expanded_dims as get_expanded_dims, index_expanded_dims as index_expanded_dims
from torch._inductor.runtime.cache_dir_utils import cache_dir as cache_dir
from torch._inductor.utils import BoxedBool as BoxedBool, InputType as InputType, count_tangents as count_tangents, fresh_cache as fresh_cache, get_all_devices as get_all_devices, is_gpu as is_gpu, should_assume_input_aligned as should_assume_input_aligned, should_use_remote_fx_graph_cache as should_use_remote_fx_graph_cache, tensor_is_aligned as tensor_is_aligned
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from torch._logging import trace_structured as trace_structured
from torch._ops import OpOverload as OpOverload
from torch._utils_internal import compile_time_strobelight_meta as compile_time_strobelight_meta
from torch.export.pt2_archive._package_weights import Weights as Weights
from torch.fx import GraphModule as GraphModule
from torch.fx.experimental.symbolic_shapes import SymExprPrinter as SymExprPrinter, free_unbacked_symbols as free_unbacked_symbols
from torch.fx.passes.fake_tensor_prop import FakeTensorProp as FakeTensorProp
from torch.monitor import _WaitCounter as _WaitCounter
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec, Protocol, TypedDict, Unpack, override

_P = ParamSpec('_P')
_T = TypeVar('_T')

def time_and_log(attr: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def log_optimus_to_scuba(*args: object, **kwargs: object) -> None: ...

class FxCompileMode(enum.Enum):
    NORMAL = 0
    SERIALIZE = 1
    SUBPROCESS = 2

def _fx_compile_mode_default() -> tuple[FxCompileMode, bool]: ...

fx_compile_mode: Incomplete
fx_compile_async: Incomplete
log: Incomplete
perf_hint_log: Incomplete
pre_grad_graphs_log: Incomplete
post_grad_graphs_log: Incomplete
static_inputs_log: Incomplete
inductor_metrics_log: Incomplete

def get_static_input_idxs(num_fixed: int) -> list[int]: ...
def record_original_output_strides(gm: GraphModule) -> None: ...
def _recursive_record_original_output_strides(gm: GraphModule) -> None: ...
def _recursive_record_user_visible_output_idxs(gm: GraphModule) -> None: ...
def _step_logger() -> Callable[..., None]: ...
@functools.cache
def _warn_tf32_disabled() -> None: ...
def _resolve_name_collision(mod: GraphModule, gm: GraphModule) -> None:
    '''
    In aot_export_module (make_fx), we create get_attr nodes with name prefix
    "_tensor_constant" and "_torchbind_obj". See Tracer.create_arg() in
    torch/fx/_symbolic_trace.py

    However, this might result in name collision if the original mod already
    has a different buffer with the same name.

    We resolve this potential name collision here by changing the target name
    with a new number post fix.
    '''
def _unlift_graph(mod: GraphModule, gm: GraphModule, graph_signature: GraphSignature) -> GraphModule: ...
def _get_subgraph_names(gm: GraphModule, skip_invoke_subgraph: bool = False) -> Generator[str, None, None]: ...
def _recursive_pre_grad_passes(gm: GraphModule, example_inputs: Sequence[InputType]) -> GraphModule: ...
def _recursive_joint_graph_passes(gm: GraphModule, skip_invoke_subgraph: bool = False) -> None: ...
def _recursive_post_grad_passes(gm: GraphModule, is_inference: bool = False) -> None: ...
def split_const_gm(gm: GraphModule, skip_constructor: bool = True, lifted_constant_names: list[str] | None = None, skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = None) -> tuple[GraphModule, dict[str, int]]:
    '''
    This function takes an GraphModule input "gm".
    The gm will be split into 2 components,
      1) const_gm, which consists the subgraph of gm that can be constant folded.
      2) gm (being inplace modified,) which returns the graph after constant folding.

    If an additional "lifted_constants" argument is passed in, we will assume the gm has
    been lifted and run the transformation accordingly.

    When a "skip_folding_node_fn" callback is passed, we will skip constant folding on
    the nodes for which the callback returns True.

    const_output_index is a mapping of corresponding node name from gm to the
    output index of const_gm.
    Returns (const_gm, const_output_index)
    '''
def is_tf32_warning_applicable(gm: GraphModule) -> bool: ...
def maybe_disable_comprehensive_padding(example_inputs: Sequence[InputType]) -> AbstractContextManager[None, None]:
    """
    For CPU backend, enable comprehensive padding causes some unit tests
    fail due to changing number of generated kernels. Skip for now.
    """
def maybe_disable_graph_partition(cpp_wrapper: bool, aot_mode: bool) -> AbstractContextManager[None, None]:
    """
    graph partition does not support cpp_wrapper and aot_mode yet.
    """
def fake_tensor_prop(gm: GraphModule, example_inputs: Sequence[InputType], force_allow_non_fake_inputs: bool = False) -> torch._subclasses.FakeTensorMode:
    """
    If we can not detect fake mode from the context of inputs, create one.

    The created fake mode will be returned.
    """
def get_patched_config_dict(config_patches: str | dict[str, Any] | None = None) -> dict[str, Any]: ...
@contextlib.contextmanager
def with_fresh_cache_if_config() -> Generator[None, None, None]: ...

class _CompileFxKwargs(TypedDict, total=False):
    cudagraphs: BoxedBool | None
    static_input_idxs: Sequence[int]
    is_backward: bool
    graph_id: int | None
    cpp_wrapper: bool
    aot_mode: bool
    is_inference: bool
    layout_opt: bool | None
    extern_node_serializer: Callable[[list[ExternKernelNode]], Any] | None
    boxed_forward_device_index: BoxedDeviceIndex | None

class _CompileFxCallable(Protocol):
    def __call__(self, gm: GraphModule, example_inputs: Sequence[InputType], **kwargs: Unpack[_CompileFxKwargs]) -> OutputCode: ...

def compile_fx_inner(gm: GraphModule, example_inputs: Sequence[InputType], **kwargs: Unpack[_CompileFxKwargs]) -> OutputCode: ...
def _compile_fx_inner(gm: GraphModule, example_inputs: Sequence[InputType], **graph_kwargs: Unpack[_CompileFxKwargs]) -> OutputCode:
    """
    Inductor API that compiles a single graph.

    If you change the argument list for this function, make sure you
    also update the call to save_args_for_compile_fx_inner below accordingly.
    """

class _FxCompileStat:
    codegen_and_compile: int
    def __repr__(self) -> str: ...

class FxCompile(ABC, metaclass=abc.ABCMeta):
    """
    An FxCompile represents a mechanism that can turn a GraphModule into an
    OutputCode.
    """
    _compile_stats: dict[type[FxCompile], _FxCompileStat]
    @abstractmethod
    def codegen_and_compile(self, gm: GraphModule, example_inputs: Sequence[InputType], inputs_to_check: Sequence[int], graph_kwargs: _CompileFxKwargs) -> OutputCode: ...
    @classmethod
    def _reset_stats(cls) -> None: ...

class _InProcessFxCompile(FxCompile):
    @override
    def codegen_and_compile(self, gm: GraphModule, example_inputs: Sequence[InputType], inputs_to_check: Sequence[int], graph_kwargs: _CompileFxKwargs) -> OutputCode:
        """
        Generates the OutputCode from the GraphModule and example_inputs.
        """

def fx_codegen_and_compile(gm: GraphModule, example_inputs: Sequence[InputType], inputs_to_check: Sequence[int], **graph_kwargs: Unpack[_CompileFxKwargs]) -> OutputCode: ...
def get_input_idxs_to_check(inputs: Sequence[InputType], static_input_idxs: Sequence[int]) -> Sequence[int]:
    """
    This function runs at compile time, and generates a list of indices for which we
    might need to do a copy to preserve alignment requirements.
    """
def cudagraphify(model: Callable[..., Any], static_input_idxs: Sequence[int] = (), *, device_index: int, stack_traces: list[str | None], is_backward: bool, is_inference: bool, constants: tuple[torch.Tensor, ...] = (), placeholders: Sequence[PlaceholderInfo] = (), mutated_input_idxs: tuple[int, ...] = ()) -> Callable[..., Any]: ...
def static_input(x: torch.Tensor) -> torch.Tensor:
    """
    Copy and input while preserving strides
    """
def index_expanded_dims_and_copy_(dst: torch.Tensor, src: torch.Tensor, expanded_dims: list[int]) -> None:
    """Index into expanded dimensions of both dst and src then copy_"""
def cudagraphify_impl(model: Callable[..., Any], inputs: list[torch.Tensor], static_input_idxs: Sequence[int] = ()) -> Callable[[list[InputType]], Any]:
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
def compile_fx_aot(model_: GraphModule, example_inputs_: list[InputType], inner_compile: _CompileFxCallable = ..., config_patches: dict[str, str] | None = None) -> list[str | Weights] | str: ...

_graph_counter: Incomplete

def fw_compiler_freezing(aot_autograd_model: GraphModule, aot_example_inputs: Sequence[InputType], dynamo_model: GraphModule, num_example_inputs: int, inner_compile: Callable[..., Any], cudagraphs: BoxedBool, graph_id: int, forward_device: BoxedDeviceIndex) -> Callable[[list[object]], Sequence[torch.Tensor]]: ...
def get_cpp_wrapper_config() -> dict[str, object]: ...
def get_cuda_device_context(gm: torch.fx.GraphModule) -> AbstractContextManager[None]:
    """
    Returns a cuda device context manager if there is a single device in the graph
    """
def compile_fx(model_: GraphModule, example_inputs_: Sequence[InputType], inner_compile: Callable[..., OutputCode] = ..., config_patches: dict[str, Any] | None = None, decompositions: dict[OpOverload, Callable[..., Any]] | None = None, ignore_shape_env: bool = False) -> Callable[[list[object]], Sequence[torch.Tensor]] | str | list[str] | Weights:
    """
    Main entry point for compiling given FX graph.  Despite the fact that this
    lives in :mod:`torch._inductor`, this function is responsible for calling
    into AOT Autograd (and we will eventually get a callback to
    ``inner_compile`` to perform actual compilation.  In other words, this
    function orchestrates end-to-end compilation for the inductor backend when
    you use :func:`torch.compile`.

    NB: This function TAKES OWNERSHIP of the input ``model_`` and can potentially
    mutate it!  Make a copy if you need to preserve the original GraphModule.
    """
def graph_returns_tuple(gm: GraphModule) -> bool:
    """True if a FX graph returns a tuple"""
def make_graph_return_tuple(gm: GraphModule, inputs: Sequence[InputType], compile_gm: Callable[..., Any]) -> Callable[..., Any]:
    """
    Mutate gm so it returns a tuple.  This is only needed for graphs
    not created by torchdynamo that return non-tuples.
    """
def handle_dynamo_export_graph(gm: GraphModule, inputs: Sequence[InputType], compile_gm: Callable[..., Any]) -> Callable[..., Any]:
    """
    `torch._dynamo.export` embeds pytrees in the FX graph codegen object,
    convert that to a normal FX graph so inductor can compile it.
    """
def _check_triton_bf16_support(graph: GraphLowering) -> None: ...
def _aoti_flatten_inputs(gm: torch.fx.GraphModule, args: list[Any] | tuple[Any, ...], kwargs: dict[str, Any] | None = None, *, options: dict[str, Any] | None = None) -> tuple[list[Any], dict[str, Any]]:
    '''
    Flatten the inputs to the graph module and return the flat inputs and options.
    Add "aot_inductor.serialized_in_spec" and "aot_inductor.serialized_out_spec" to the options.
    '''
