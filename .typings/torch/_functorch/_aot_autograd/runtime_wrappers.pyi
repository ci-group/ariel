import contextlib
import torch
from .. import config as config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata as run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base as gen_alias_from_base
from .input_output_analysis import compute_overlapping_inputs as compute_overlapping_inputs, create_synthetic_base_metadata as create_synthetic_base_metadata, remove_dupe_metadata as remove_dupe_metadata
from .logging_utils import describe_input as describe_input, format_guard_bug_msg as format_guard_bug_msg, track_graph_compiling as track_graph_compiling
from .schemas import AOTConfig as AOTConfig, InputAliasInfo as InputAliasInfo, MemoryFormatMeta as MemoryFormatMeta, MutationType as MutationType, OutputType as OutputType, PlainTensorMeta as PlainTensorMeta, SubclassCreationMeta as SubclassCreationMeta, SubclassMeta as SubclassMeta, TensorAlias as TensorAlias, ViewAndMutationMeta as ViewAndMutationMeta
from .subclass_utils import requires_subclass_dispatch as requires_subclass_dispatch, runtime_unwrap_tensor_subclasses as runtime_unwrap_tensor_subclasses, wrap_tensor_subclasses as wrap_tensor_subclasses
from .traced_function_transforms import aot_dispatch_subclass as aot_dispatch_subclass
from .utils import call_func_at_runtime_with_args as call_func_at_runtime_with_args, make_boxed_func as make_boxed_func, partial_flatten_asdict as partial_flatten_asdict, strict_zip as strict_zip
from _typeshed import Incomplete
from collections.abc import Generator
from dataclasses import dataclass, field
from torch import Tensor as Tensor
from torch._dynamo.callback import CallbackTrigger as CallbackTrigger, callback_handler as callback_handler
from torch._dynamo.utils import CompileEventLogger as CompileEventLogger, dynamo_timed as dynamo_timed, get_metrics_context as get_metrics_context
from torch._guards import CompileContext as CompileContext, DuplicateInputs as DuplicateInputs, TracingContext as TracingContext, compile_context as compile_context, detect_fake_mode as detect_fake_mode, tracing as tracing
from torch._prims_common import CUDARngStateHelper as CUDARngStateHelper
from torch._subclasses import FakeTensor as FakeTensor
from torch.fx.experimental._backward_state import BackwardState as BackwardState
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass
from typing import Any, Callable

zip = strict_zip

class CompilerWrapper:
    """
    A wrapper around the inputs and outputs to the compiler_fn. We separate these into two parts:

    1. The prologue, which edits the input to the compiler_fn(flat_fn, flat_args, etc)
    2. The epilogue, which edits the outputs of the compiler_fn (compiled_fn, real arguments)

    Each wrapper below should be implemented as a CompilerWrapper, so that we can facilitate
    caching on the compiled output, and re-wrapping the output via epilogues.
    Extra metadata that is needed to compute pre or post compile can be passed in via attributes.
    """
    def pre_compile(self, flat_fn, flat_args: list[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> tuple[Callable, list[Tensor], ViewAndMutationMeta]:
        """
        Process the inputs to the compiler_fn. You can pass in extra metadata via kwargs.
        Args:
        flat_fn: The function to compile
        flat_args: Metadata from example inputs of the function to compile
        aot_config: AOTConfig passed in at compile time
        fw_metadata: ViewAndMutationMeta generated from flat_fn and flat_args
        """
    def post_compile(self, compiled_fn, aot_config, *, runtime_metadata) -> Callable:
        """
        Given an output of the compiler, wrap it with information received from prologue.
        Args:
        compiled_fn: Callable after calling compiler_fn
        aot_config: AOTConfig after calling prologue
        runtime_metadata: ViewAndMutationMeta after calling all wrappers's pre_compile steps.
        Example:

        def wrapped_compiled_fn(args):
            # do something with args, aot_config, fw_metadata
            return compiled_fn(args)

        return wrapped_compiled_fn
        """

@dataclass
class RuntimeWrapper(CompilerWrapper):
    indices_of_inps_to_detach: list[int]
    trace_joint: bool
    disable_amp: bool
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

class NoopAliasHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

def _unwrap_tensoralias(x): ...
def _identity(x): ...

class AliasOfInputHandler:
    base_idx: Incomplete
    unwrap_out: Incomplete
    requires_grad: Incomplete
    functional_tensor: Incomplete
    replay_views: Incomplete
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

class IsInputHandler:
    base_idx: Incomplete
    unwrap_out: Incomplete
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

class AliasOfIntermediateHandler:
    _unwrap_aliased_base_tensor: Incomplete
    base_idx: Incomplete
    unwrap_out: Incomplete
    requires_grad: Incomplete
    functional_tensor: Incomplete
    replay_views: Incomplete
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

_HANDLER_MAP: Incomplete

def make_output_handler(info, runtime_metadata, trace_joint): ...
def maybe_mark_dynamic_helper(t: torch.Tensor, dims: set[int]): ...
def _should_disable_saved_tensors_hooks(): ...
def _create_runtime_wrapper(compiled_fn, *, runtime_metadata: ViewAndMutationMeta, indices_of_inps_to_detach: list[int], trace_joint: bool, keep_input_mutations: bool, disable_amp: bool): ...

@dataclass
class FunctionalizedRngRuntimeWrapper(CompilerWrapper):
    return_new_outs: bool = ...
    def pre_compile(self, flat_fn, flat_args, aot_config, *, fw_metadata) -> tuple[Callable, list[Tensor], ViewAndMutationMeta]: ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...
    def _functionalized_rng_runtime_epilogue(self, metadata: ViewAndMutationMeta, outs, offset_index): ...

@dataclass
class FakifiedOutWrapper(CompilerWrapper):
    out_metas: list[torch.Tensor] = field(default_factory=list)
    fwd_output_strides: list[list[int] | None] | None = ...
    needs_post_compile: bool = ...
    def pre_compile(self, fw_module, flat_args, aot_config, *, fw_metadata) -> tuple[Callable, list[Tensor], ViewAndMutationMeta]: ...
    def _compute_output_meta_with_inductor_strides(self): ...
    def set_fwd_output_strides(self, fwd_output_strides) -> None: ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class AOTDispatchSubclassWrapper(CompilerWrapper):
    trace_joint: bool
    fw_only: Callable | None
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None
    def pre_compile(self, flat_fn, flat_args: list[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta): ...
    def post_compile(self, compiled_fn, _aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class EffectTokensWrapper(CompilerWrapper):
    def post_compile(self, compiled_fn, _aot_config, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class AOTDedupeWrapper(CompilerWrapper):
    keep_arg_mask: list[bool] = field(default_factory=list)
    add_dupe_map: list[int] = field(default_factory=list)
    old_input_metadata: list[InputAliasInfo] = field(default_factory=list)
    needs_post_compile: bool = ...
    def remove_dupe_args(self, args): ...
    def add_dupe_args(self, args): ...
    def pre_compile(self, flat_fn, flat_args: list[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> tuple[Callable, list[Tensor], ViewAndMutationMeta]: ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class AOTSyntheticBaseWrapper(CompilerWrapper):
    trace_joint: bool
    needs_post_compile: bool = ...
    aliased_arg_idx_with_metadata_mutations: list[int] = field(default_factory=list)
    old_input_info = ...
    def pre_compile(self, flat_fn, flat_args: list[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> tuple[Callable, list[Tensor], ViewAndMutationMeta]: ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

def merge_view_inputs(aot_config: AOTConfig, fwd_inputs: list[Any], mutated_input_info: list[InputAliasInfo], *, is_inference: bool) -> tuple[list[Any], list[int | tuple[int, torch.Tensor]] | None]: ...

@dataclass
class AutogradLazyBackwardCompileInfo:
    bw_module: Callable
    placeholder_list: list[Any]
    saved_context: TracingContext | None
    saved_compile_context: CompileContext | None

@dataclass
class CachedAutogradLazyBackwardCompileInfo:
    bw_module_fn: Callable

def _raise_if_functorch_active(): ...
def _backward_prologue_functional(ctx_saved_tensors, ctx_symints, metadata, maybe_subclass_metadata, *flat_args): ...
def initialize_rng_states(num_rng: int, graphsafe_idx: int, fwd_rng_states: list[torch.Generator], bwd_rng_states: list[torch.Generator]):
    """
    Initialize the cudagraph safe rng states.

    Initialization of rng states should have a few properties:
    - the initialization for each rng state should be independent
    - the initialization should be deterministic
    - the initialization should be based off current rng state, so that independent graphs do not
    have equal rng behavior

    We defer initialization of rng states until runtime because compilation is wrapped
    with preserve_rng_states. Seed initialization should advance the rng states so consecutive compilations
    do not give equal randomness.
    """
def _backward_epilogue_functional(metadata, maybe_subclass_metadata, out, *, make_subclass_override=None): ...
def coerce_to_expected_memory_format(x: torch.Tensor, memory_format: MemoryFormatMeta): ...
@contextlib.contextmanager
def _disable_saved_tensors_hooks() -> Generator[None]: ...

class AOTDispatchAutograd:
    @staticmethod
    def process_runtime_tangent(x, meta: PlainTensorMeta | SubclassCreationMeta): ...
    @staticmethod
    def post_compile(compiled_fw_func, compiled_bw_func, maybe_subclass_meta: SubclassMeta | None, num_symints_saved_for_bw_: int, backward_state_indices: list[int], disable_amp: bool, indices_of_inps_to_detach: list[int], lazy_backward_info: AutogradLazyBackwardCompileInfo | CachedAutogradLazyBackwardCompileInfo | None, aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta, try_save_cache_entry: Callable | None): ...

@dataclass
class DebugAssertWrapper(CompilerWrapper):
    flat_requires_grad: list[bool | None] = field(default_factory=list)
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

def pre_compile(wrappers: list[CompilerWrapper], flat_fn: Callable, flat_args: list[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta) -> tuple[Callable, list[Tensor], ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function and arguments.
    Mutates wrappers in place.
    """
def post_compile(wrappers: list[CompilerWrapper], compiled_fn: Callable, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta) -> tuple[Callable, ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function. Should be called after pre_compile()
    """
def make_runtime_safe(fw_metadata: ViewAndMutationMeta, maybe_subclass_meta: SubclassMeta | None):
    """
    Calls make_runtime_safe on all ViewAndMutationMetas.
    Modifies both arguments. Allows ViewAndMutationMetas to
    be safely cached in AOTAutogradCache.
    """
