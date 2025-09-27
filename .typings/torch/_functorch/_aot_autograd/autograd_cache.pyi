import abc
import contextlib
import functools
import torch
from .runtime_wrappers import AOTDispatchAutograd as AOTDispatchAutograd, AOTDispatchSubclassWrapper as AOTDispatchSubclassWrapper, CachedAutogradLazyBackwardCompileInfo as CachedAutogradLazyBackwardCompileInfo, CompilerWrapper as CompilerWrapper, FunctionalizedRngRuntimeWrapper as FunctionalizedRngRuntimeWrapper, RuntimeWrapper as RuntimeWrapper, SubclassMeta as SubclassMeta, post_compile as post_compile
from .schemas import AOTAutogradCacheInfo as AOTAutogradCacheInfo, AOTConfig as AOTConfig, ViewAndMutationMeta as ViewAndMutationMeta
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch._dynamo.precompile_context import PrecompileCacheArtifact as PrecompileCacheArtifact, PrecompileContext as PrecompileContext
from torch._dynamo.trace_rules import torch_non_c_binding_in_graph_functions as torch_non_c_binding_in_graph_functions
from torch._dynamo.utils import CompileEventLogger as CompileEventLogger, chromium_event_log_active as chromium_event_log_active, counters as counters, dynamo_timed as dynamo_timed
from torch._functorch import config as config
from torch._inductor.codecache import BypassFxGraphCache as BypassFxGraphCache, FxGraphCache as FxGraphCache, FxGraphCachePickler as FxGraphCachePickler, FxGraphHashDetails as FxGraphHashDetails, GuardedCache as GuardedCache, _ident as _ident, add_ephemeral_timeout_increase_for_distributed as add_ephemeral_timeout_increase_for_distributed, create_cache as create_cache, extract_tensor_metadata_for_cache_key as extract_tensor_metadata_for_cache_key, sha256_hash as sha256_hash, write_atomic as write_atomic
from torch._inductor.compile_fx import _CompileFxKwargs as _CompileFxKwargs
from torch._inductor.cudagraph_utils import BoxedDeviceIndex as BoxedDeviceIndex
from torch._inductor.output_code import CompiledFxGraph as CompiledFxGraph, CompiledFxGraphConstants as CompiledFxGraphConstants, OutputCode as OutputCode
from torch._inductor.remote_cache import JsonDataTy as JsonDataTy, RemoteCache as RemoteCache
from torch._inductor.runtime.runtime_utils import cache_dir as cache_dir
from torch._inductor.utils import BoxedBool as BoxedBool, should_use_remote_fx_graph_cache as should_use_remote_fx_graph_cache
from torch._logging import LazyString as LazyString
from torch._utils_internal import log_cache_bypass as log_cache_bypass
from torch.compiler._cache import CacheArtifact as CacheArtifact, CacheArtifactFactory as CacheArtifactFactory, CacheArtifactManager as CacheArtifactManager
from torch.fx.experimental.symbolic_shapes import hint_int as hint_int
from torch.fx.node import Node as Node
from torch.utils._triton import has_triton_package as has_triton_package
from typing import Any, Callable, Generic, TypeVar
from typing_extensions import override

log: Incomplete

class BypassAOTAutogradCache(Exception): ...
class FXGraphCacheMiss(BypassAOTAutogradCache): ...

def should_use_remote_autograd_cache(): ...
def should_use_local_autograd_cache(): ...
def check_node_safe(node: Node):
    '''
    Checks that the node only uses supported operators. We are starting with very
    conservative cacheability constraints, and incrementally adding more support as we expand.

    [Note: AOTAutograd Cacheability checks]
    - Our cache key is computed from the FX graph produced by Dynamo and the input example values
    - A node is "safe" if the same cache key results in a compiled artifact that has the same behavior
        (i.e, the set of inputs that go into our cache key is sufficient to distinguish its behavior)

    To accomplish this safety check, we consider the following functions to be safe:
        - Public functions under modules torch, torch.functional, and torch.nn.functional: these are
        allowed in the graph by dynamo, so we can assume they are safe to cache.
        - method calls on base tensor types
        - Any call_module that dynamo deemed safe to allow AOTAutograd to trace
        - Non callable nodes, such as placeholder, output, get_attr

    The test suite test_aot_autograd_cache.py::AOTAutogradCachePicklerTests tries its best to fully cover/specify this behavior.
    '''
def check_cacheable(gm: torch.fx.GraphModule):
    """
    Checks that the graph module only uses supported operators
    """
def check_metadata_cacheable(metadata: ViewAndMutationMeta):
    """
    When view replay is turned on, we bypass autograd cache if
    the output is aliased.
    """

class AOTAutogradCacheDetails(FxGraphHashDetails):
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """
    aot_config: Incomplete
    grad_enabled: Incomplete
    disable_amp: Incomplete
    deterministic_algorithms: Incomplete
    autograd_config: Incomplete
    saved_tensors_hooks_fx_wrap_cache_hashes: tuple[list[str], list[str]]
    def __init__(self, gm: torch.fx.GraphModule, example_inputs, aot_config: AOTConfig, fx_config: _CompileFxKwargs) -> None: ...

class AOTAutogradCachePickler(FxGraphCachePickler):
    dispatch_table: dict
    def __init__(self, gm: torch.fx.GraphModule) -> None: ...
    def _reduce_aot_config(self, aot_config: AOTConfig):
        """
        Reduce the config to a stable key for caching.
        """
    def _reduce_tensor(self, tensor):
        """
        Reduce the tensor to a stable key for caching.
        """

def autograd_cache_key(gm: torch.fx.GraphModule, example_inputs, config: AOTConfig, fx_config: _CompileFxKwargs) -> tuple[str, list[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """
TOut = TypeVar('TOut', bound=OutputCode)

class InductorOutput(ABC, Generic[TOut], metaclass=abc.ABCMeta):
    """
    Class representing a single inductor output
    """
    @abstractmethod
    def pre_save(self) -> None: ...
    @abstractmethod
    def load(self, example_inputs) -> TOut: ...
    @abstractmethod
    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut: ...

@dataclass
class CompiledFxGraphLoadable(InductorOutput[CompiledFxGraph]):
    """
    A full compiled fx graph that doesn't need to lookup the FxGraphCache
    to run
    """
    result: CompiledFxGraph
    def pre_save(self) -> None: ...
    example_inputs = ...
    def load(self, example_inputs) -> CompiledFxGraph: ...
    def post_compile(self, result: CompiledFxGraph, fx_config: _CompileFxKwargs) -> CompiledFxGraph: ...

@dataclass
class FxGraphCacheLoadable(InductorOutput[CompiledFxGraph]):
    fx_graph_cache_info: tuple[str, list[str]]
    fx_graph_guard_expr: str | None
    def pre_save(self) -> None: ...
    def _is_backward(self) -> bool: ...
    example_inputs = ...
    constants = ...
    def load(self, example_inputs) -> CompiledFxGraph: ...
    def post_compile(self, result: CompiledFxGraph, fx_config: _CompileFxKwargs) -> CompiledFxGraph:
        """
        Called after FXGraphCacheLoadable.load, mutates fx_config
        """

@dataclass
class CompiledForward(FxGraphCacheLoadable):
    """
    Cacheable entry for a forward function
    """
    def _is_backward(self) -> bool: ...

@dataclass
class GenericCompiledBackward(InductorOutput[TOut], metaclass=abc.ABCMeta):
    backward_state_indices: list[int]
    num_symints_saved_for_bw_: int

@dataclass
class CompiledBackward(GenericCompiledBackward[CompiledFxGraph], FxGraphCacheLoadable):
    """
    Cacheable entry for a forward function
    """
    def _is_backward(self) -> bool: ...
    def post_compile(self, result: CompiledFxGraph, fx_config: _CompileFxKwargs) -> CompiledFxGraph: ...

class BundledCompiledForward(CompiledFxGraphLoadable): ...

@dataclass
class BundledCompiledBackward(GenericCompiledBackward[CompiledFxGraph], CompiledFxGraphLoadable):
    def post_compile(self, result: CompiledFxGraph, fx_config: _CompileFxKwargs) -> CompiledFxGraph: ...

@dataclass
class SerializedGraphModule:
    fn: Callable[[dict[Any, Any], str], torch.nn.Module]
    args: tuple[Any, ...]
    def __init__(self, gm: torch.fx.GraphModule) -> None: ...
    def deserialize(self) -> torch.fx.GraphModule: ...

def serialize_graph_module(gm: torch.fx.GraphModule) -> SerializedGraphModule: ...
TForward = TypeVar('TForward', bound=InductorOutput)
TBackward = TypeVar('TBackward', bound=GenericCompiledBackward)

@dataclass
class GenericAOTAutogradCacheEntry(Generic[TForward, TBackward]):
    """A single entry into the cache, genericized by Forward and Backward types.

    A TForward is always an InductorOutput of some sort, which represents the
    forward graph of the compile.
    A TBackward is an InductorOutput + metadata about the backward, useful for specific
    backward-only wrappers. This type is encapsulated by GenericCompiledBackward.

    Each AOTAutogradCacheEntry is essentially parameterized by 1. the method of loading
    from the cache (either Bundled or UnBundled), and 2. The type of the output. For now,
    the only type of output we support is Python Wrapper output, i.e. OutputCode.CompiledFxGraph,
    but the same technique works for C++ wrapper code; we'd just add an extra InductorOutput type.
    """
    compiled_fw: TForward
    compiled_bw: TBackward | None
    aot_joint_graph_str: str | None
    aot_forward_graph_str: str | None
    aot_backward_graph_str: str | None
    runtime_metadata: ViewAndMutationMeta
    dispatch_wrappers: list[CompilerWrapper]
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None
    indices_of_inps_to_detach: list[int]
    forward_time_taken_ns: int
    backward_time_taken_ns: int
    sanitized_aot_config: AOTConfig
    guards_expr: str | None
    serialized_bw_module: SerializedGraphModule | None
    def pre_save(self) -> None:
        """
        Perform any preparations to make the cache entry ready for serialization.
        """
    def wrap_post_compile(self, args: list[torch.Tensor], aot_config: AOTConfig, fx_config: _CompileFxKwargs) -> Callable:
        """
        This function takes a cache entry and carefully reconstructs the original callable
        that AOTAutograd returned the first time it was run. It does this by running the various
        post compile steps that AOTAutograd runs on its compiled artifact after running the fw/bw compilers.

        In the inference path, this consists of the Subclass, FunctionalzedRngRuntime, and RuntimeWrappers.
        In the autograd path, this consists of AOTAutogradDispatch.post_compile.

        The steps here should match exactly the steps that are run in aot_dispatch_base and aot_dispatch_autograd.

        Notably absent from the cached path are:
        - DebugAssertWrapper
        - FakifiedOutWrapper

        Which we'll handle separately later on, if necessary.
        """

class AOTAutogradCacheEntry(GenericAOTAutogradCacheEntry[CompiledForward, CompiledBackward]):
    """
    Regular AOTAutogradCacheEntry: saves the forward/backward FxGraphCache keys
    and looks them up in FxGraphCache on load
    """
class BundledAOTAutogradCacheEntry(GenericAOTAutogradCacheEntry[BundledCompiledForward, BundledCompiledBackward]):
    """
    AOTAutogradCacheEntry where we save the entire CompiledFxGraph instead
    of relying on cache keys from FxGraphCache
    """

@contextlib.contextmanager
def sanitize_gm_for_cache(gm: torch.fx.GraphModule):
    """
    Clears a few fields in a dynamo supplied Graph Module that are not stable between graph inputs, but don't
    affect inductor or aotdispatch correctness.

    These fields **can** be used by code calling into aotdispatch (namely, dynamo), so we can't null them out completely.

    To ensure that these fields are not accessed by inductor or aotdispatch, we clear them during AOTAutogradCache.load,
    and then put them back before returning. This way, we generate a cache key based off of a canonical graph
    without these fields, and also guarantee they aren't used to affect the cache's output.
    """

class AOTAutogradCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None: ...
    @override
    @staticmethod
    def type(): ...

class BundledAOTAutogradCacheArtifact(PrecompileCacheArtifact[Callable]):
    @override
    @staticmethod
    def type(): ...
    @override
    def after_deserialization(self) -> Callable: ...

class AOTAutogradCache(GuardedCache[GenericAOTAutogradCacheEntry]):
    """
    Caches the results of running AOTAutograd. This class mostly handles the save and load logic, whereas
    AOTAutogradCacheEntry handles the wrapping/unwrapping logic.

    Cache Inputs (AOTAutogradCacheDetails)
    - AOTAutogradCache takes in the following inputs, which are analogous to inputs given
        to AOTAutograd by dynamo:
        - A fx graph module generated by dynamo
        - A list of args, which consists of:
            - Symint inputs to the graph, generated by dynamo
            - The **real tensor** inputs, which inductor uses for cudagraphs
            - Notably, the real tensor inputs don't have symints in their metadata.
        AOTAutograd then retraces those real tensor arguments into FakeTensors later during execution.
        - A set of global configurations that affect AOTAutograd or Inductor behavior.

    It then generates a cache key given these values. Notably, this means AOTAutogradCache currently
    specializes on the sizes and strides of the real tensor inputs when dynamic shapes are turned on.
    In a later PR, we'll likely generate the cache key based on the FakeTensors AOTAutograd generates
    based on the real tensor inputs, which can contain symints.

    # Cache Outputs (AOTAutogradCacheEntry)
    - AOTAutogradCache caches the following values:
        - The compiled forward and backward functions from inductor, via keys to the FXGraphCache
        - Metadata to reconstruct the AOTModule from the compiled inductor artifacts
        - See AOTAutogradCacheEntry for more info

    [Note: Caching guards generated by AOTAutograd and Inductor]
    AOTAutograd and inductor both can introduce new guards to the shape environment. FXGraphCache saves guards with each
    compiled graph inductor generates. On a cache hit, AOTAutograd reloads the compiled forward and backward functions
    from FXGraphCache, giving it new symint arguments from the input args.
    FXGraphCache uses those symints and its saved guards to repopulate the ShapeEnv with guards.
    **No new guards are generated into the shape env after inductor finishes compiling**, so the guards
    saved by inductor are sufficient for correctness for both AOTAutograd and Inductor's caches.
    """
    @staticmethod
    def clear() -> None:
        """Clear the cache"""
    @staticmethod
    def load(dispatch_and_compile: Callable, mod: torch.fx.GraphModule | torch._dynamo.utils.GmWrapper, args, aot_config: AOTConfig, cudagraphs: BoxedBool, boxed_forward_device_index: BoxedDeviceIndex | None, local: bool, remote: bool) -> Callable:
        """
        Load a result from the cache, and reconstruct a runtime wrapper around the object
        """
    @classmethod
    def generate_guards_expression(cls, cache_info: AOTAutogradCacheInfo) -> str | None: ...
    @classmethod
    def _get_tmp_dir(cls) -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
    @classmethod
    def _get_tmp_dir_for_key(cls, key) -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
    @staticmethod
    def evaluate_guards(guard_expr: str, hints: list[int] | list[torch.SymInt]): ...
    @staticmethod
    def _lookup(key: str, local: bool, remote: bool, args: list[Any], cache_info: dict[str, Any], aot_config: AOTConfig | None) -> GenericAOTAutogradCacheEntry | None:
        """Given a key generated by AOTAutogradCachePickler, look up its location in the cache."""
    @staticmethod
    def _write_to_local_cache(key: str, content: bytes):
        """Write an entry to the local cache."""
    @staticmethod
    def save(key: str, entry: GenericAOTAutogradCacheEntry, remote: bool):
        """Save a single entry into the cache."""
    @staticmethod
    @functools.cache
    def get_remote_cache() -> RemoteCache[JsonDataTy] | None:
        """
        Attempts to load the remote cache, returns None on error.
        """
    @staticmethod
    def make_entry(compiled_fw_func: CompiledFxGraph, compiled_bw_func: CompiledFxGraph | None, aot_joint_graph_str: str | None, aot_forward_graph_str: str | None, aot_backward_graph_str: str | None, runtime_metadata: ViewAndMutationMeta, dispatch_wrappers: list[CompilerWrapper], maybe_subclass_meta: SubclassMeta | None, num_fw_outs_saved_for_bw: int | None, indices_of_inps_to_detach: list[int], forward_time_taken_ns: int, backward_time_taken_ns: int, sanitized_aot_config: AOTConfig, guards_expr: str | None, backward_state_indices: list[int] | None, num_symints_saved_for_bw: int | None, serialized_bw_module: SerializedGraphModule | None) -> GenericAOTAutogradCacheEntry: ...
