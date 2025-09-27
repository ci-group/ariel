import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from . import config as config
from ._aot_autograd.autograd_cache import AOTAutogradCache as AOTAutogradCache, autograd_cache_key as autograd_cache_key, should_use_local_autograd_cache as should_use_local_autograd_cache, should_use_remote_autograd_cache as should_use_remote_autograd_cache
from ._aot_autograd.collect_metadata_analysis import run_functionalized_fw_and_collect_metadata as run_functionalized_fw_and_collect_metadata
from ._aot_autograd.functional_utils import _check_if_mutation_can_be_in_graph as _check_if_mutation_can_be_in_graph, are_all_mutations_hidden_from_autograd as are_all_mutations_hidden_from_autograd, are_all_mutations_under_no_grad_or_inference_mode as are_all_mutations_under_no_grad_or_inference_mode, assert_functional_graph as assert_functional_graph, from_fun as from_fun, gen_alias_from_base as gen_alias_from_base, has_data_mutation as has_data_mutation, has_metadata_mutation as has_metadata_mutation, is_fun as is_fun, sync_functional_tensor as sync_functional_tensor, to_fun as to_fun
from ._aot_autograd.input_output_analysis import compute_overlapping_inputs as compute_overlapping_inputs, create_graph_signature as create_graph_signature, create_synthetic_base_metadata as create_synthetic_base_metadata, remove_dupe_metadata as remove_dupe_metadata
from ._aot_autograd.jit_compile_runtime_wrappers import aot_dispatch_autograd as aot_dispatch_autograd, aot_dispatch_base as aot_dispatch_base, aot_dispatch_export as aot_dispatch_export
from ._aot_autograd.logging_utils import callback_set as callback_set, describe_input as describe_input, format_guard_bug_msg as format_guard_bug_msg, get_aot_compilation_context as get_aot_compilation_context, get_aot_graph_name as get_aot_graph_name, get_graph_being_compiled as get_graph_being_compiled, graph_being_compiled as graph_being_compiled, model_name as model_name, nth_graph as nth_graph, set_model_name as set_model_name, setup_stacktrace_preservation_hooks as setup_stacktrace_preservation_hooks, track_graph_compiling as track_graph_compiling
from ._aot_autograd.runtime_wrappers import AOTDedupeWrapper as AOTDedupeWrapper, AOTSyntheticBaseWrapper as AOTSyntheticBaseWrapper
from ._aot_autograd.schemas import AOTConfig as AOTConfig, BackwardSignature as BackwardSignature, FQN as FQN, GraphInputName as GraphInputName, GraphOutputName as GraphOutputName, GraphSignature as GraphSignature, InputAliasInfo as InputAliasInfo, MutationType as MutationType, OutputAliasInfo as OutputAliasInfo, OutputType as OutputType, SubclassCreationMeta as SubclassCreationMeta, SubclassMeta as SubclassMeta, TensorAlias as TensorAlias, ViewAndMutationMeta as ViewAndMutationMeta
from ._aot_autograd.subclass_utils import requires_subclass_dispatch as requires_subclass_dispatch, unwrap_tensor_subclasses as unwrap_tensor_subclasses, unwrap_tensor_subclasses_with_indices_to_original as unwrap_tensor_subclasses_with_indices_to_original, wrap_tensor_subclasses as wrap_tensor_subclasses, wrap_tensor_subclasses_maybe_joint as wrap_tensor_subclasses_maybe_joint
from ._aot_autograd.traced_function_transforms import aot_dispatch_subclass as aot_dispatch_subclass, create_functional_call as create_functional_call, create_functionalized_fn as create_functionalized_fn, create_functionalized_rng_ops_wrapper as create_functionalized_rng_ops_wrapper, create_joint as create_joint, fn_input_mutations_to_outputs as fn_input_mutations_to_outputs, fn_prepped_for_autograd as fn_prepped_for_autograd
from ._aot_autograd.utils import KNOWN_TYPES as KNOWN_TYPES, _get_autocast_states as _get_autocast_states, _get_symint_hints as _get_symint_hints, call_func_at_runtime_with_args as call_func_at_runtime_with_args, create_tree_flattened_fn as create_tree_flattened_fn, make_boxed_compiler as make_boxed_compiler, make_boxed_func as make_boxed_func, maybe_to_fresh_input as maybe_to_fresh_input, normalize_as_list as normalize_as_list, partial_flatten_asdict as partial_flatten_asdict, root_module_when_exporting_non_strict as root_module_when_exporting_non_strict, strict_zip as strict_zip
from .partitioners import default_partition as default_partition
from _typeshed import Incomplete
from collections.abc import KeysView, Sequence
from contextlib import contextmanager
from torch import Tensor as Tensor
from torch._decomp.decompositions_for_rng import PhiloxStateTracker as PhiloxStateTracker, rng_decompositions as rng_decompositions
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._dynamo import compiled_autograd as compiled_autograd
from torch._dynamo.utils import CompileEventLogger as CompileEventLogger, dynamo_timed as dynamo_timed, preserve_rng_state as preserve_rng_state, set_feature_use as set_feature_use
from torch._guards import detect_fake_mode as detect_fake_mode
from torch._inductor.cudagraph_utils import BoxedDeviceIndex as BoxedDeviceIndex
from torch._inductor.output_code import OutputCode as OutputCode
from torch._inductor.utils import BoxedBool as BoxedBool, InputType as InputType
from torch._subclasses import FakeTensor as FakeTensor, FakeTensorMode as FakeTensorMode
from torch.fx.experimental.proxy_tensor import _pytree_subclasses_that_lose_info as _pytree_subclasses_that_lose_info, make_fx as make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass
from typing import Any, Callable, Protocol, TypeVar

static_inputs_log: Incomplete
zip = strict_zip
AOT_COUNTER: Incomplete
aot_autograd_decompositions: Incomplete
FakifiedFlatArgs: Incomplete
TOutputCode = TypeVar('TOutputCode', bound=OutputCode)

class AOTDispatchCompiler(Protocol):
    """
    Represents a fw or bw_compiler passed to AOTAutograd.
    """
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: Sequence[InputType]) -> Any: ...

class SerializableAOTDispatchCompiler(AOTDispatchCompiler):
    """
    Represents an AOTDispatchCompiler that returns an OutputCode, and is
    therefore cacheable. SerializableAOTDispatchCompiler always return an OutputCode.
    A _CompileFxCallable usually gets converted into an AOTDispatchCompiler after binding all of
    the kwargs in _CompileFxKwargs.
    """
    output_code_ty: Incomplete
    compiler_fn: Incomplete
    def __init__(self, output_code_ty: type[TOutputCode], compiler_fn: Callable[[torch.fx.GraphModule, Sequence[InputType]], TOutputCode]) -> None: ...
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: Sequence[InputType]) -> OutputCode: ...

def process_inputs(flat_args: list[Any], aot_config: AOTConfig, fake_mode: FakeTensorMode, shape_env: ShapeEnv | None, ignore_shape_env: bool = False) -> FakifiedFlatArgs: ...
def construct_fake_mode(flat_args: list[Any], aot_config: AOTConfig) -> tuple[FakeTensorMode, ShapeEnv | None]: ...
def create_aot_dispatcher_function(flat_fn, fake_flat_args: FakifiedFlatArgs, aot_config: AOTConfig, fake_mode: FakeTensorMode, shape_env: ShapeEnv | None) -> tuple[Callable, ViewAndMutationMeta]: ...
def _create_aot_dispatcher_function(flat_fn, fake_flat_args: FakifiedFlatArgs, aot_config: AOTConfig, fake_mode: FakeTensorMode, shape_env: ShapeEnv | None) -> tuple[Callable, ViewAndMutationMeta]:
    """
    Traces the forward and backward graphs of the attr:`flat_fn` to generate a
    joint graph. The joint graph is an Fx graph with Aten ops. Please refer to
    the tracing mechanism to understand the graph capturing details.

    The joint graph is then passed through attr:`partition_fn` to isolate the
    forward and backward portions, which are then respectively compiled via the
    provided attr:`fw_compiler` and attr:`bw_compiler`.

    The resulting compiled forward and backward graphs are then wrapped up in a
    ``torch.autograd.Function`` object.

    The calling convention here is that the first aot_config.num_params_buffers
    inputs in flat_args are parameters and buffers, and the rest are inputs.

    We use this to assume that parameters/buffer's shapes don't change.

    Note: this function is used both by aot_function and aot_export (controlled by aot_config.is_export)
        When aot_config.is_export is True, we return an FX graph + metadata
        When aot_config.is_export is False, we return an ordinary runtime function
    """
def aot_function(fn: Callable, fw_compiler: Callable, bw_compiler: Callable | None = None, partition_fn: Callable = ..., decompositions: dict | None = None, num_params_buffers: int = 0, keep_inference_input_mutations: bool = False, inference_compiler: Callable | None = None, *, dynamic: bool = False, enable_log: bool = True) -> Callable:
    """
    Traces the forward and backward graph of :attr:`fn` using torch dispatch
    mechanism, and then compiles the generated forward and backward graphs
    through :attr:`fw_compiler` and :attr:`bw_compiler`.

    :func:`aot_function` traces the forward and backward graph ahead of time,
    and generates a joint forward and backward graph.  :attr:`partition_fn` is
    then used to separate out forward and backward graphs. The partitioner
    function can be used to perform optimizations such as recomputation. One can
    set `decompositions` dictionary to decompose the operators into a sequence
    of core or simpler operators supported by the backend compilers.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Callable): A Python function that takes one ore more arguments. Must
            return one or more Tensors.
        fw_compiler (Callable): A Python function that accepts an Fx graph with
            Aten ops and input args, and returns a Callable that semantically is
            equivalent to the input Fx graph.
        bw_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph.  Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
        partition_fn (Callable): A Python function that takes a joint forward
            and backward graph, and partitions it into separate forward and
            backward graphs.
        decompositions (Dict): A dictionary to define the decomposition of
            larger Aten ops into simpler or core Aten ops.
        inference_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph. inference_compiler is invoked
            if no autograd is needed. Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
    Returns:
        Returns a ``Callable`` that retains the eager behavior of the original
        :attr:`fn`, but with forward and backward graph compiled via
        :attr:`fw_compile` and :attr:`bw_compile`.

    A simple example usage of :func:`aot_function` is as follows. This example
    will print the forward and backward graphs of the function ``fn``

        >>> fn = lambda x : x.sin().cos()
        >>> def print_compile_fn(fx_module, args):
        >>>     print(fx_module)
        >>>     return fx_module
        >>> aot_fn = aot_function(fn, print_compile_fn)
        >>> x = torch.randn(4, 5, requires_grad=True)
        >>> aot_fn(x)
    """
def aot_module(mod: nn.Module, *args, **kwargs) -> nn.Module:
    """
    Traces the forward and backward graph of :attr:`mod` using torch dispatch
    tracing mechanism. It is wrapper function, that underneath uses
    :func:`aot_function` to perform tracing and compilation.

    :func:`aot_module` lifts the parameters and buffers of ``nn.Module`` as inputs
    to a new callable which is then compiled through :func:`aot_function`.

    .. warning::
        This API is experimental and likely to change.

    Args:
        mod (Callable): A ``nn.Module`` module.
        args : args to be passed to :func:`aot_function`
        kwargs : kwargs to be passed to :func:`aot_function`

    Returns:
        Returns a ``nn.Module`` that retains the eager behavior of the original
        :attr:`mod`, but with forward and backward graph compiled.

    """
def _try_get_metadata_from_dynamo(mod: torch.nn.Module, param_keys: KeysView[str], full_args_num: int) -> tuple[list[torch._guards.Source] | None, list[int]]:
    """
    Metadata is forwarded from Dynamo to AOTDispatch via special fields on GraphModule.
    We first verify that `mod` does come from Dynamo, then we handle cases where
    metadata might be missing.

    Returns:
        aot_autograd_arg_pos_to_source: used to dedup params and their guards
        static_input_indices: used to identify static inputs for cudagraphs
    """
def aot_module_simplified(mod: nn.Module, args, fw_compiler: AOTDispatchCompiler, bw_compiler: AOTDispatchCompiler | None = None, partition_fn: Callable = ..., decompositions: dict | None = None, keep_inference_input_mutations: bool = False, inference_compiler: AOTDispatchCompiler | None = None, cudagraphs: BoxedBool | None = None, boxed_forward_device_index: BoxedDeviceIndex | None = None, ignore_shape_env: bool = False) -> nn.Module:
    """
    This is the simplified or low overhead version of aot_module. For frontends
    like TorchDynamo, the input functions/modules to AOT are static and have
    unpacked inputs/outputs. This gives us an opportunity to remove the
        (1) pytree overhead to parse inputs/outputs,
        (2) AOT Autograd cache,
        (3) Reading of params/buffers in every forward call

    :func:`aot_module_simplified` removes these overheads.
    """
def aot_export_module(mod: nn.Module, args, *, decompositions: dict | None = None, trace_joint: bool, output_loss_index: int | None = None, pre_dispatch: bool = False, dynamic_shapes: bool | None = None, kwargs=None) -> tuple[torch.fx.GraphModule, GraphSignature]:
    """
    This function takes in a module, and returns:
    (1) an FX graph that can be exported
    (2) some metadata about the graph

    If `trace_joint=True` we will return a joint graph of the forward + backward.

    The traced FX graph will have the following properties compared to the original module:
    (1) Inputs and outputs to the module will be pytree-flattened
    (2) Parameters and buffers on the module will be lifted into graph inputs,
        graph_inputs = (*parameters, *buffers, *user_inputs)
    (3) The graph will be fully functionalized
    (4) Any input mutations will be converted into additional outputs in the graph,
        meaning whoever calls this graph is responsible for applying the mutations
        back to the original inputs.
    (5) If is_joint is provided the graph will return parameter gradients in addition to user outputs.
        The graph output will look like:
        graph_outputs = (*updated_inputs, *user_outputs, *param_gradients)

    There are also several restrictions on what modules can use this API. In particular:
    (1) If trace_joint is specified, we expect the loss function to be **fused**
        into the module forward. One of the outputs to the forward must be a scalar loss,
        which is specified with `output_loss_index`.
        All other outputs to the forward are presumed to not require gradients.
    (2) This API cannot capture optimizers (although in theory we could build an API for this).
    (3) Metadata mutations on params/buffers/inputs are banned.
    (4) Data mutations on anything that requires gradients are banned (parameters)
    (5) If an input is mutated, it is not allowed to alias any other inputs.
    (6) Parameters must not be duplicated.
    """
def aot_export_joint_simple(func: Callable, args, *, trace_joint: bool, num_params_buffers: int = 0, decompositions: dict | None = None) -> torch.fx.GraphModule:
    '''
    A simplified version of export. Used by higher order operators.

    This function makes a high-level "no calling convention changes" guarantee:
    - If no inputs require grad (so we export an inference graph),
      there are *no* calling convention change between the exported graph, and "func".
    - If at least one input requires grad (so we trace out and export a joint fw-bw graph),
      Then if you were partition the graph into a separate forward and backward graph,
      The forward graph will have no calling convention changes compared to "func".

    The above also relies on some strong restrictions around which functions this API accepts:
    (1) `args` cannot contain any pytrees (they must have been pytree_flattened already)
    (2) `func` cannot mutate any inputs
    (3) The outputs of `func` cannot alias any inputs.

    Note: this function is only lightly tested today. It will probably be tested more heavily by higher order ops.
    '''
def _aot_export_function(func: Callable, args, *, num_params_buffers: int = 0, decompositions: dict | None = None, no_tangents: bool = False, pre_dispatch: bool = False, dynamic_shapes: bool | None = None, kwargs=None) -> tuple[torch.fx.GraphModule, ViewAndMutationMeta, pytree.TreeSpec, pytree.TreeSpec]: ...
@contextmanager
def _detect_attribute_assignment(mod: torch.nn.Module): ...
compiled_function = aot_function
compiled_module = aot_module
