import dataclasses
import functools
import hashlib
import pickle
import torch
from .compile_fx import _CompileFxKwargs as _CompileFxKwargs
from .cpp_builder import BuildOptionsBase as BuildOptionsBase
from .graph import GraphLowering as GraphLowering
from .ir import ChoiceCaller as ChoiceCaller
from .output_code import CompiledFxGraph as CompiledFxGraph, CompiledFxGraphConstants as CompiledFxGraphConstants, OutputCode as OutputCode
from .remote_cache import JsonDataTy as JsonDataTy, RemoteCache as RemoteCache, create_cache as create_cache
from .runtime import autotune_cache as autotune_cache
from .runtime.autotune_cache import AutotuneCacheBundler as AutotuneCacheBundler
from .runtime.hints import HalideInputSpec as HalideInputSpec, HalideMeta as HalideMeta
from .runtime.triton_heuristics import CachingAutotuner as CachingAutotuner
from .triton_bundler import TritonBundler as TritonBundler
from .utils import InputType as InputType
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Generator, KeysView, Sequence
from concurrent.futures import Future
from ctypes import CDLL, c_void_p
from functools import lru_cache, partial
from pathlib import Path
from torch import SymInt as SymInt, Tensor as Tensor
from torch._dynamo.exc import SkipFrame as SkipFrame
from torch._dynamo.utils import CompileEventLogger as CompileEventLogger, counters as counters, dynamo_timed as dynamo_timed
from torch._inductor import config as config, exc as exc, metrics as metrics
from torch._inductor.codegen.common import custom_backend_passes as custom_backend_passes, init_backend_registration as init_backend_registration
from torch._inductor.codegen.cuda import cuda_env as cuda_env
from torch._inductor.codegen.rocm.compile_command import rocm_compile_command as rocm_compile_command, rocm_compiler as rocm_compiler
from torch._inductor.compile_worker.utils import in_toplevel_process as in_toplevel_process
from torch._inductor.cpp_builder import CppBuilder as CppBuilder, CppOptions as CppOptions, CppTorchDeviceOptions as CppTorchDeviceOptions, _LINKER_SCRIPT as _LINKER_SCRIPT, _TORCH_PATH as _TORCH_PATH, _set_gpu_runtime_env as _set_gpu_runtime_env, _transform_cuda_paths as _transform_cuda_paths, convert_cubin_to_obj as convert_cubin_to_obj, get_compiler_version_info as get_compiler_version_info, get_ld_and_objcopy as get_ld_and_objcopy, get_name_and_dir_from_output_file_path as get_name_and_dir_from_output_file_path, normalize_path_separator as normalize_path_separator
from torch._inductor.cpu_vec_isa import pick_vec_isa as pick_vec_isa
from torch._inductor.custom_graph_pass import CustomGraphModulePass as CustomGraphModulePass, CustomGraphPass as CustomGraphPass, CustomGraphPassType as CustomGraphPassType
from torch._inductor.fb.utils import log_global_cache_errors as log_global_cache_errors, log_global_cache_stats as log_global_cache_stats, log_global_cache_vals as log_global_cache_vals, use_global_cache as use_global_cache
from torch._inductor.freezing_utils import has_frozen_params as has_frozen_params, is_frozen_param as is_frozen_param
from torch._inductor.runtime.compile_tasks import _reload_python_module as _reload_python_module
from torch._inductor.runtime.runtime_utils import cache_dir as cache_dir, default_cache_dir as default_cache_dir
from torch._inductor.utils import ALIGN_BYTES as ALIGN_BYTES, clear_on_fresh_cache as clear_on_fresh_cache, is_linux as is_linux, is_windows as is_windows
from torch._logging import trace_structured as trace_structured
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor, TensorMetadata as TensorMetadata, extract_tensor_metadata as extract_tensor_metadata
from torch._utils_internal import log_cache_bypass as log_cache_bypass
from torch.compiler._cache import CacheArtifact as CacheArtifact, CacheArtifactFactory as CacheArtifactFactory, CacheArtifactManager as CacheArtifactManager
from torch.export.pt2_archive._package_weights import TensorProperties as TensorProperties, Weights as Weights
from torch.export.pt2_archive.constants import CUSTOM_OBJ_FILENAME_PREFIX as CUSTOM_OBJ_FILENAME_PREFIX
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv, has_hint as has_hint, hint_int as hint_int
from torch.utils._ordered_set import OrderedSet as OrderedSet
from types import ModuleType
from typing import Any, Callable, Generic, NoReturn, TypeVar
from typing_extensions import Self, override

T = TypeVar('T')
_IS_WINDOWS: Incomplete
LOCK_TIMEOUT: int
output_code_log: Incomplete
log: Incomplete

def use_re_build() -> bool:
    """
    Use for CUTLASS compilation only right now.
    """
def get_cpp_wrapper_cubin_path_name() -> str: ...
def get_kernel_bin_format(device: str) -> str: ...
@functools.cache
def get_global_cache_path_impl(global_cache_dir: str) -> Path | None: ...

class CacheBase:
    @staticmethod
    @functools.cache
    def get_system() -> dict[str, Any]: ...
    @staticmethod
    @clear_on_fresh_cache
    @functools.cache
    def get_local_cache_path() -> Path: ...
    @staticmethod
    def get_global_cache_path() -> Path | None: ...
    system: Incomplete
    def __init__(self) -> None: ...
    def get_local_cache(self) -> dict[str, Any]: ...
    def update_local_cache(self, local_cache: dict[str, Any]) -> None: ...

class LocalCache(CacheBase):
    def lookup(self, *keys: str) -> dict[str, Any] | None: ...
    def set_value(self, *keys: str, value: Any) -> None: ...

class PersistentCache(CacheBase):
    @functools.cache
    def get_global_cache(self) -> dict[str, Any]: ...
    def lookup(self, choices: list[ChoiceCaller], op: str, inputs: str, benchmark: Callable[[Any], dict[ChoiceCaller, float]] | None) -> dict[ChoiceCaller, float]:
        """
        Check to see if we have benchmarked the given choice callers. For each
        choice caller:

            1. Check global_cache[op][inputs][choice][precision], return benchmark if cached.
            2. Check local_cache[op][inputs][choice][precision], return benchmark if cached.
            3. If benchmark is not None:
                a. `max_autotune_gemm=True`: benchmark the choice, update
                    local_cache[op][inputs][choice], and return the benchmark.
                b. `max_autotune_gemm=False`: don't benchmark the choice, return nothing.
        """

def get_lock_dir() -> str: ...
def sha256_hash(data: bytes) -> str: ...
def code_hash(code: str | bytes, extra: str | bytes = '') -> str: ...
def get_path(basename: str, extension: str, specified_dir: str = '') -> tuple[str, str, str]: ...
def get_hash(content: str | bytes, extra: str = '', hash_type: str = 'code') -> str: ...
def write(content: str | bytes, extension: str, extra: str = '', hash_type: str = 'code', specified_dir: str = '', key: str | None = None) -> tuple[str, str]: ...
def write_text(text: str) -> str:
    """
    Write the `text` to a file and return the path computed based on the hash.
    """
def write_atomic(path_: str, content: str | bytes, make_dirs: bool = False, encode_utf_8: bool = False) -> None: ...

@dataclasses.dataclass
class TensorMetadataAndValues:
    """
    TensorMetadata plus the elements as a list of raw values.
    Used for hashing inlined constants.
    """
    tensor_metadata: TensorMetadata
    values: list[Any]

def _ident(x: T) -> T: ...
def extract_tensor_metadata_for_cache_key(t: Tensor) -> TensorMetadata:
    """
    Extracts the tensor metadata and removes fields of the TensorMetadata
    that are not needed for caching
    """

class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """
    _stream: Incomplete
    dispatch_table: Incomplete
    fast: bool
    def __init__(self, gm: torch.fx.GraphModule, has_user_defined_triton_kernels: bool = False) -> None:
        """
        Create an FX graph pickler. If include_non_inlined=True, then pickling will
        include the _values_ for all Tensors. (Note that any tensors are constants
        attached as attributes to the GraphModule). Otherwise, pickling will include
        only the metadata for these tensors.
        """
    def _reduce_fake_tensor(self, t: Tensor) -> tuple[Callable[[T], T], tuple[TensorMetadata]]:
        """
        Custom reducer to pickle FakeTensors.
        """
    def _reduce_tensor(self, t: Tensor) -> tuple[Callable[[T], T], tuple[TensorMetadata | TensorMetadataAndValues]]:
        """
        Custom reducer to pickle Tensors.  If we see tensors, we know they're constants
        stored as attributes on the GraphModule.
        """
    def _reduce_symint(self, s: SymInt) -> tuple[Callable[[T], T], tuple[str]]:
        """
        Custom reducer to pickle SymInts.
        """
    def _reduce_unsupported(self, s: Any) -> NoReturn:
        """
        Custom reducer to handle any objects that we don't support and therefore
        raise to bypass caching.
        """
    def _reduce_graph_module(self, gm: torch.fx.GraphModule) -> tuple[Any, tuple[dict[str, Any], str]]:
        """
        Custom reducer for graph module to handle irrelevant data for user
        defined triton kernels
        Essentially what we are doing here is a huge hack where user defined
        triton kernel contain a dynamo time side table and the arguments to the
        call_function are indices into this side table. These arguments are not
        for hashing purposes since we included the source code into the cache
        key and the numbers are prone to give false negatives due to ordering.
        """
    def dumps(self, obj: Any) -> bytes:
        """
        Pickle an object and return a byte string.
        """
    def get_hash(self, obj: Any) -> str:
        """
        Serialize an object and return a hash of the bytes.
        """
    def debug_lines(self, inp: FxGraphHashDetails) -> list[str]:
        """
        Get a printable string describing in more detail all the attributes
        comprising an object. Useful for debugging when one graph hashes
        to a different value than another.
        """

def build_code_hash(roots: list[str] | None, prefix: str, hasher: hashlib._Hash) -> None: ...
def torch_key_cache(func: Callable[[], bytes]) -> Callable[[], bytes]:
    """
    This function is a reimplementation of functools.lru_cache with a
    set function that allows prepopulating the cache.
    """
@torch_key_cache
def torch_key() -> bytes:
    """
    Compute a key that contains relevant information about torch source files
    """
def get_inductor_root() -> str: ...

@dataclasses.dataclass
class OrderedSetHolder:
    """
    See FxGraphHashDetails. Holds a sorted list to support stable hashing
    of set kwargs.
    """
    items: list[Any]

class BypassFxGraphCache(Exception):
    """
    Exception to indicate that the FxGraphCache should be bypassed.
    """

class FxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """
    EXCLUDED_KWARGS: Incomplete
    gm: Incomplete
    example_inputs: Incomplete
    cache_key_tag: Incomplete
    fx_kwargs: dict[str, object]
    user_defined_triton_source: list[Any]
    inputs_to_check: Incomplete
    default_cuda_device_index: Incomplete
    deterministic_algorithms_settings: Incomplete
    cuda_matmul_settings: Incomplete
    torch_version: Incomplete
    system_info: Incomplete
    inductor_config: Incomplete
    post_grad_custom_pre_pass: Incomplete
    post_grad_custom_post_pass: Incomplete
    _pre_fusion_custom_pass: Incomplete
    _fuse_ddp_communication_passes: Incomplete
    custom_backend_passes: Incomplete
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: Sequence[InputType], fx_kwargs: _CompileFxKwargs, inputs_to_check: Sequence[int]) -> None: ...
    def _get_custom_pass_detail_unsafe(self, custom_pass: Any) -> Any | None: ...
    def _get_custom_pass_detail(self, custom_pass: CustomGraphPassType | CustomGraphModulePass) -> Any | None: ...

def compiled_fx_graph_hash(gm: torch.fx.GraphModule, example_inputs: Sequence[InputType], fx_kwargs: _CompileFxKwargs, inputs_to_check: Sequence[int]) -> tuple[str, list[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """
def add_ephemeral_timeout_increase_for_distributed(time_saved_ns: int) -> int:
    """
    Ephemerally increases the NCCL timeout when compiling for a distributed job
    Returns amount of seconds increased
    """

class GuardedCache(Generic[T]):
    """
    Mixin for caches that have guards associated with their entries.
    """
    @classmethod
    def _get_tmp_dir_for_key(cls, _key: str) -> str: ...
    @classmethod
    def iterate_over_candidates(cls, local: bool, remote_cache: RemoteCache[JsonDataTy] | None, key: str) -> Generator[tuple[T, bytes], None, None]: ...
    @classmethod
    def find_guarded_entry(cls, key: str, local: bool, remote_cache: RemoteCache[JsonDataTy] | None, evaluate_guards: Callable[[str, list[int] | list[torch.SymInt]], bool], hints: list[int]) -> tuple[T | None, bytes | None, dict[str, str]]:
        """
        Find the first cache entry in iterate_over_candidates that passes `evaluate_guards`.

        Args:
            key: The cache key to look up
            local: Whether to check the local cache
            remote_cache: The remote cache to check, if any
            evaluate_guards: Function that evaluates whether a guard passes the check,
                given a list of hint values and the guard expression.
            hints: List of symint hints paired with evaluate_guards

        Returns:
            A tuple of (graph, pickled_content) if found, or (None, None) if not found
        """
    @classmethod
    def _filter_backed_symints(cls, inputs: Sequence[InputType]) -> list[torch.SymInt]:
        """
        Get the backed SymInt objects from the input list. Note that we can never
        have guards that depend on unbacked symint.
        """
    @classmethod
    def _get_shape_env(cls) -> ShapeEnv | None:
        """
        Helper to get the shape env from the tracing context.
        """

class InductorCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None: ...
    @override
    @staticmethod
    def type() -> str: ...

class FxGraphCache(GuardedCache[CompiledFxGraph]):
    """
    Supports caching and reusing compiled Fx graphs.

    The overall strategy is as follows:
    - This cache stores entries on disk. When saving an entry, we can't
      serialize callables (that could be C++, Triton, etc.), so we serialize
      their own disk cache location. We then recreate the compiled artifact
      after fetching from disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See FxGraphCachePickler.
    - Among the metadata we store, we also include a guards expression that's
      appropriate for validating any symbols for Tensor arguments that have
      symbolic bounds. On cache lookup then, we evaluate those guards in the
      current context to validate that a cached entry can be served.
    - A given graph could have multiple compiled versions, corresponding to
      different sets of guards. Therefore, we store cache entries in the form:
          <temp dir>/<fx graph hash>/<serialized metadata>
    - On lookup, we compute the key from the graph details, iterate over all
      leaf files in the corresponding subdirectory, deserialize the entry, and
      evaluate its guards expression. If the evaluation succeeds, we have a
      cache hit. If it fails, we compile the graph and store a new entry.
    - Finally, on a cache hit, we need to make sure any guards that would
      have been created during compilation are added to the current context.
    """
    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
    @classmethod
    def _get_tmp_dir_for_key(cls, key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
    @staticmethod
    def cache_hit_post_compile(graph: CompiledFxGraph, cache_info: dict[str, Any], constants: CompiledFxGraphConstants) -> tuple[CompiledFxGraph | None, dict[str, Any]]:
        """
        Cache specific post compile steps that need to run if we find a graph in the cache
        This includes putting bundled triton artifacts in the right place,
        reloading the PyCodeCache artifact, etc.

        These don't always happen (i.e. on a cache miss, so they are in a separate function from
        CompiledFxGraph.post_compile)
        """
    @staticmethod
    def _lookup_graph(key: str, example_inputs: Sequence[InputType], local: bool, remote_cache: RemoteCache[JsonDataTy] | None, constants: CompiledFxGraphConstants, evaluate_guards: Callable[[str, list[int] | list[torch.SymInt]], bool] | None = None) -> tuple[CompiledFxGraph | None, dict[str, Any]]:
        """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledFxGraph object. On a miss, return None.
        `constants` tracks a list of constants, or a way to obtain the list of constants
        associated with a given cache entry
        `evaluate_guards` allows AOTAutogradCache and other callers to customize
        what constitutes a guard success. Normally, a guard hit happens if
        `shape_env.evaluate_guards_expression` returns True.
        """
    @staticmethod
    def _write_to_local_cache(key: str, content: bytes) -> None: ...
    @staticmethod
    def _save_graph(key: str, compiled_graph: OutputCode, example_inputs: Sequence[InputType], local: bool, remote_cache: RemoteCache[JsonDataTy] | None) -> None:
        """
        Store a serialized CompiledFxGraph on disk.
        """
    @staticmethod
    def _check_for_hop(gm: torch.fx.GraphModule) -> None: ...
    @staticmethod
    def _check_can_cache(gm: torch.fx.GraphModule) -> None:
        """
        Check some conditions that would preclude caching and raise BypassFxGraphCache
        to bypass in case caching is not possible.
        """
    @staticmethod
    def prepare_key(gm: torch.fx.GraphModule, example_inputs: Sequence[InputType], fx_kwargs: _CompileFxKwargs, inputs_to_check: Sequence[int], remote: bool) -> tuple[tuple[str, list[str]] | None, dict[str, Any]]:
        """
        Checks that the inductor input is cacheable, then computes
        and returns the cache key for the input.
        Returns (key_info, cache_info) where:
        - key_info is (hash_key, debug_lines), and
        - cache_info will contain debug info in the event of BypassFxGraphCache.

        NB: It is possible to have this function return a union instead. But
        I personally believe it is more annoying/difficult to read in that format.
        """
    @staticmethod
    def get_remote_cache() -> RemoteCache[JsonDataTy] | None:
        """
        Attempts to load the remote cache, returns None on error.
        """
    @staticmethod
    def load_with_key(key: str, debug_lines: list[str], example_inputs: Sequence[InputType], local: bool, remote_cache: RemoteCache[JsonDataTy] | None, is_backward: bool, constants: CompiledFxGraphConstants, evaluate_guards: Callable[[str, list[int] | list[torch.SymInt]], bool] | None = None) -> tuple[CompiledFxGraph | None, dict[str, Any]]:
        """
        Lookup the graph with the given key, and return results and metadata.
        Doesn't do any logging on its own, because AOTAutograd handles a cache miss
        differently from FXGraphCache.
        """
    @staticmethod
    def clear() -> None:
        """
        Clear out the on-disk cache.
        """

@functools.cache
def split_aot_inductor_output_path(path: str) -> tuple[str, str]:
    """Returns the path where the AOT Inductor compiled kernels are stored."""

class CudaKernelParamCache:
    cache: dict[str, dict[str, Any]]
    cache_clear: Incomplete
    @classmethod
    def set(cls, key: str, params: dict[str, str | None], cubin: str, bin_type: str, asm: str | None = None, asm_type: str | None = None) -> None: ...
    @classmethod
    def get(cls, key: str) -> dict[str, Any] | None: ...
    @classmethod
    def get_keys(cls) -> KeysView[str]: ...

class AotCodeCompiler:
    """
    Compile AOT Inductor generated code.
    """
    @classmethod
    def compile(cls, graph: GraphLowering, wrapper_code: str, kernel_code: str, serialized_extern_kernel_nodes: str | None, *, device_type: str, additional_files: list[str]) -> list[str | Weights] | str:
        """
        Returns the .so path, or returns a list of files that were generated if
        config.aot_inductor.package=True.
        """

_libgomp: CDLL | None

def custom_op_wrapper(op: str, *args: Any) -> list[c_void_p] | c_void_p | None: ...

_HEADER_DIR: Incomplete
_HEADER_LOCK_DIR: Incomplete

@functools.cache
def _precompile_header(header: str, hashable_cmd_line: str, **compile_command: Any) -> str: ...
def _get_cpp_prefix_header(device: str) -> str | None: ...
def _get_cpp_wrapper_header(device: str, aot_mode: bool = False) -> str:
    """Given a device type (and optionally whether we're in AOT Inductor mode), returns
    the path to the cpp_wrapper header file to be precompiled."""

class CppCodeCache:
    """Compiles and caches C++ libraries.  Users of this class supply the source code to
    be compiled, while compilation flags are set by CppBuilder."""
    cache: dict[str, Callable[[], CDLL | ModuleType]]
    cache_clear: Incomplete
    cpp_compile_command_flags: dict[str, Any]
    @staticmethod
    def _load_library_inner(path: str, key: str) -> CDLL | ModuleType: ...
    @classmethod
    def _load_library(cls, path: str, key: str) -> CDLL | ModuleType: ...
    @classmethod
    def _get_uncompiled_header(cls, device: str) -> str | None:
        """
        Given a device type, returns the path to a CPP header file to be precompiled.
        """
    @classmethod
    def load_async(cls, main_code: str, device_type: str = 'cpu', submit_fn: Any = None, extra_flags: Sequence[str] = (), optimized_code: str | None = None) -> Any:
        """Compile and load a C++ library.  Returns a callable that returns the loaded
        library."""
    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Any: ...

def _worker_compile_cpp(lock_path: str, cpp_builders: Sequence[CppBuilder]) -> None: ...

class CppPythonBindingsCodeCache(CppCodeCache):
    cache: dict[str, Callable[[], CDLL | ModuleType]]
    cache_clear: Incomplete
    cpp_compile_command_flags: Incomplete
    entry_function: str
    call_entry_function: str
    extra_parse_arg: str
    suffix_template: Incomplete
    @classmethod
    def _load_library_inner(cls, path: str, key: str) -> ModuleType: ...
    @classmethod
    def _get_uncompiled_header(cls, device: str) -> str | None: ...
    @classmethod
    def load_pybinding_async(cls, argtypes: Sequence[str], main_code: str, device_type: str = 'cpu', num_outputs: int = -1, submit_fn: Any = None, extra_flags: Sequence[str] = (), kernel_code: str | None = None) -> Any:
        '''
        Wrap a C++ function in fast Python bindings.

        Args:
            argtypes: The types of args to ENTRY_FUNCTION(), e.g. ["float*", "long"]
            main_code: C++ source code containing ENTRY_FUNCTION().  Will be built at
                -O3 if kernel_code is None (to maximize performance in any kernels that
                are present), or -O1 otherwise (to minimize compile time).
            kernel_code: If present, C++ source code that will be built at -O3 and
                linked to main_code.

        Returns:
            A python version of ENTRY_FUNCTION()
        '''
    @classmethod
    def load_pybinding(cls, *args: Any, **kwargs: Any) -> Any: ...

class CppWrapperCodeCache(CppPythonBindingsCodeCache):
    cache: dict[str, Callable[[], CDLL | ModuleType]]
    cache_clear: Incomplete
    cpp_compile_command_flags: Incomplete
    entry_function: str
    call_entry_function: str
    extra_parse_arg: Incomplete
    @classmethod
    def _get_uncompiled_header(cls, device: str) -> str | None: ...

class HalideCodeCache(CppPythonBindingsCodeCache):
    cache: dict[str, Callable[[], ModuleType | CDLL]]
    cache_clear: Incomplete
    _standalone_runtime_path: str | None
    prefix: Incomplete
    glue_template_cpp: Incomplete
    glue_template_cuda: Incomplete
    standalone_runtime_cuda_init: Incomplete
    @classmethod
    def _codegen_buffer(cls, name: str, arg: HalideInputSpec, cuda: bool) -> list[str]: ...
    @classmethod
    def _codegen_glue(cls, meta: HalideMeta, headerfile: object) -> str: ...
    @classmethod
    @functools.cache
    def config_hash(cls) -> str: ...
    @staticmethod
    def _search_for_file(suffix: str, errmsg: str) -> str: ...
    @staticmethod
    @functools.cache
    def find_libautoschedule(name: str) -> str: ...
    @staticmethod
    @functools.cache
    def find_header(name: str) -> str: ...
    @classmethod
    def generate_halide_async(cls, meta: HalideMeta, source_code: str, submit_fn: Any = None) -> Callable[[], Any]: ...
    @classmethod
    def generate_halide(cls, *args: Any, **kwargs: Any) -> Callable[[], Any]: ...
    @classmethod
    def build_standalone_runtime(cls) -> str: ...
    @classmethod
    def _get_uncompiled_header(cls, device: str) -> str | None:
        """Header precompiling is currently disabled for halide."""

def _worker_task_halide(lockfile: str, jobs: list[partial[Any]]) -> None: ...
def touch(filename: str) -> None: ...

class PyCodeCache:
    modules: list[ModuleType]
    modules_no_attr: dict[str, ModuleType]
    linemaps: dict[str, list[tuple[Any, ...]]]
    @classmethod
    def write(cls, source_code: str, extra: str = '') -> tuple[str, str]: ...
    @classmethod
    def load(cls, source_code: str, extra: str = '') -> ModuleType: ...
    @classmethod
    def load_by_key_path(cls, key: str, path: str, linemap: list[tuple[int, str]] | None = None, attrs: dict[str, Any] | None = None) -> ModuleType: ...
    @classmethod
    def cache_clear(cls, purge: bool = False) -> None:
        """
        Clear the in-memory module cache. If purge=True, also delete all the
        corresponding on-disk source files.
        """
    @classmethod
    @functools.cache
    def stack_frames_for_code(cls, path: str, lineno: int) -> list[dict[str, Any]] | None: ...

def _load_triton_kernel_from_source(kernel_name: str, source_code: str) -> CachingAutotuner: ...
def _cuda_compiler() -> str | None: ...
def _cutlass_path() -> str: ...
def _cutlass_paths() -> list[str]: ...
def _clone_cutlass_paths(build_root: str) -> list[str]: ...
def _cutlass_include_paths() -> list[str]: ...
@torch_key_cache
def cutlass_key() -> bytes:
    """
    Compute a key representing the state of the CUTLASS library.

    Note: OSS and fbcode will have different keys.
    """
def _cuda_lib_options() -> list[str]:
    """
    Util function for CUTLASS backend to find the correct CUDA libraries.
    """
def _nvcc_host_compiler_options() -> list[str]: ...
def _nvcc_arch_as_compile_option() -> str: ...
def _nvcc_compiler_options() -> list[str]: ...
def cuda_compile_command(src_files: list[str], dst_file: str, dst_file_ext: str, extra_args: list[str] | None = None) -> str: ...

class DLLWrapper:
    """A wrapper for a dynamic library."""
    lib_path: Incomplete
    is_open: bool
    DLL: Incomplete
    def __init__(self, lib_path: str) -> None: ...
    def close(self) -> None: ...
    def _dlclose(self) -> None: ...
    def __getattr__(self, name: str) -> Callable[..., None]: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: Any) -> None: ...
    def __del__(self) -> None: ...

@lru_cache
def binary_error_path(output_path: str) -> str:
    """
    standard format for the error path
    """

class CUDACodeCache:
    """
    A cache for managing the compilation and loading of CUDA source code specifically for CUTLASS.
    This class handles writing source code to files, compiling them into shared objects, and caching
    the results to avoid redundant compilations. It also manages error handling and logging for the
    compilation process.
    """
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str
        error_json: str | None = ...
    cache: dict[str, CacheEntry]
    aot_kernels_o: list[str]
    _SOURCE_CODE_SUFFIX: str
    @staticmethod
    def cache_clear() -> None: ...
    @staticmethod
    def get_kernel_binary_remote_cache(caching_enabled: bool, caching_available: bool) -> Any | None:
        """
        Get or create the class instance of the CUTLASSKernelBinaryRemoteCache.

        Args:
            caching_enabled: Whether binary remote caching is enabled
            caching_available: Whether we're in fbcode environment

        Returns:
            CUTLASSKernelBinaryRemoteCache: The class instance of the kernel binary remote cache
        """
    @classmethod
    def write(cls, source_code: str, dst_file_ext: str) -> tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """
    @classmethod
    def compile(cls, source_code: str, dst_file_ext: str, extra_args: list[str] | None = None) -> tuple[str, str, str]:
        """
        Compiles CUDA source_code into a file with dst_file_ext extension.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
    @classmethod
    def load(cls, source_code: str, dst_file_ext: str) -> tuple[DLLWrapper, str, str]:
        """
        Compiles source code and loads the generated .so file.
        Returns a tuple of DLLWrapper, hash_key, source_code_path
        """
    @classmethod
    def _record_cuda_compile_error(cls, error_str: str, key: str, cmd_parts: list[str], input_path: str, output_path: str, binary_remote_cache: Any = None) -> None: ...

class ROCmCodeCache:
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str
    cache: dict[str, CacheEntry]
    aot_kernels_o: list[str]
    _SOURCE_CODE_SUFFIX: str
    _logged_compiler_version: bool
    @staticmethod
    def cache_clear() -> None: ...
    @classmethod
    def write(cls, source_code: str, dst_file_ext: str) -> tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """
    @classmethod
    def compile(cls, source_code: str, dst_file_ext: str, extra_args: list[str] | None = None) -> tuple[str, str, str]:
        """
        Compiles source_code into a file with dst_file_ext extension,
        using the compile command specific for the ROCm platform.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
    @classmethod
    def load(cls, source_code: str, dst_file_ext: str) -> tuple[DLLWrapper, str, str]:
        """
        Compiles source code and loads the generated .so file.
        Returns a tuple of DLLWrapper, hash_key, source_code_path
        """

class CodeCacheFuture:
    def result(self) -> Callable[..., Any]: ...

class LambdaFuture(CodeCacheFuture):
    result_fn: Incomplete
    future: Incomplete
    def __init__(self, result_fn: Callable[..., Any], future: Future[Any] | None = None) -> None: ...
    def result(self) -> Callable[..., Any]: ...

class StaticAutotunerFuture(CodeCacheFuture):
    """
    A statically launchable CachingAutotuner, loaded from TritonBundler
    """
    static_autotuner: Incomplete
    reload_kernel_from_src: Callable[[], Any] | None
    def __init__(self, static_autotuner: CachingAutotuner) -> None: ...
    def result(self) -> CachingAutotuner: ...
