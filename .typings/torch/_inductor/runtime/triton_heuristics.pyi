import dataclasses
from . import triton_helpers as triton_helpers
from ..triton_bundler import TritonBundler as TritonBundler
from ..utils import prefix_is_reduction as prefix_is_reduction, triton_version_uses_attrs_dict as triton_version_uses_attrs_dict
from .autotune_cache import AutotuneCache as AutotuneCache
from .benchmarking import benchmarker as benchmarker
from .coordinate_descent_tuner import CoordescTuner as CoordescTuner
from .hints import AutotuneHint as AutotuneHint, DeviceProperties as DeviceProperties, HeuristicType as HeuristicType, ReductionHint as ReductionHint, TRITON_MAX_BLOCK as TRITON_MAX_BLOCK, TRITON_MAX_RSPLIT as TRITON_MAX_RSPLIT, TileHint as TileHint, _NUM_THREADS_PER_WARP as _NUM_THREADS_PER_WARP
from .runtime_utils import ceildiv as ceildiv, compilation_callback as compilation_callback, conditional_product as conditional_product, create_bandwidth_info_str as create_bandwidth_info_str, dynamo_timed as dynamo_timed, get_first_attr as get_first_attr, get_max_y_grid as get_max_y_grid, get_num_bytes as get_num_bytes, next_power_of_2 as next_power_of_2, triton_cache_dir as triton_cache_dir, triton_config_to_hashable as triton_config_to_hashable, triton_hash_to_path_key as triton_hash_to_path_key, validate_triton_config as validate_triton_config
from .static_cuda_launcher import StaticallyLaunchedCudaKernel as StaticallyLaunchedCudaKernel
from .triton_compat import ASTSource as ASTSource, CompiledKernel as CompiledKernel, Config as Config, GPUTarget as GPUTarget, HAS_WARP_SPEC as HAS_WARP_SPEC, KernelInterface as KernelInterface, OutOfResources as OutOfResources, PTXASError as PTXASError, autograd_profiler as autograd_profiler, cc_warp_size as cc_warp_size, knobs as knobs, triton as triton
from _typeshed import Incomplete
from collections.abc import Container
from torch._dynamo.utils import set_feature_use as set_feature_use
from torch._guards import CompileId as CompileId
from torch._prims_common import compute_required_storage_length as compute_required_storage_length
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable, Generic, Literal, TypeVar

class NoTritonConfigsError(RuntimeError): ...
LauncherType = Any
_KernelType = CompiledKernel | StaticallyLaunchedCudaKernel
_T = TypeVar('_T', bound=_KernelType)
log: Incomplete

def get_total_reduction_numel(numels: dict[str, int]) -> int: ...
def autotune_hints_to_configs(hints: OrderedSet[AutotuneHint], size_hints, block_size: int, device_props: DeviceProperties) -> list[Config]:
    """
    AutotuneHints can be attached to the metadata of triton kernels for providing
    suggestions about what to try for autotuning. One reason to do this is if there are
    some configs that are only useful in specific scenarios, in which case we can avoid
    wasting compile time on autotuning unless we know we are in one of those scenarios.

    Based on those hints, this function will generate a list of additional autotuning
    configs to try.
    """
def disable_pointwise_autotuning(inductor_meta): ...
def _dump_launch_params(args, kwargs, launcher, kernel_name, grid) -> None: ...
def check_autotune_cache(configs: list[Config], filename: str | None, inductor_meta: dict[str, Any]) -> tuple[list[Config], AutotuneCache | None, dict[str, Any]]:
    """
    Given a list of configs, checks autotune cache and return metadata
    """

class CachingAutotuner(KernelInterface):
    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """
    fn: Incomplete
    device_props: DeviceProperties
    triton_meta: Incomplete
    inductor_meta: Incomplete
    save_cache_hook: Incomplete
    mutated_arg_names: Incomplete
    reset_to_zero_arg_names: Incomplete
    optimize_mem: Incomplete
    configs: Incomplete
    heuristic_type: Incomplete
    custom_kernel: Incomplete
    cuda_kernel_saved: bool
    autotune_cache_info: Incomplete
    compile_results: list[CompileResult[_KernelType]]
    launchers: list[LauncherType]
    lock: Incomplete
    size_hints: Incomplete
    coordesc_tuner: Incomplete
    filename: Incomplete
    kernel_hash: str
    precompile_time_taken_ns: int
    autotune_time_taken_ns: int
    dump_launch_params: Incomplete
    triton_interpret: Incomplete
    compile_id: CompileId | None
    is_backward: bool
    def __init__(self, fn, triton_meta, configs, save_cache_hook, mutated_arg_names: list[str], optimize_mem, heuristic_type, size_hints=None, inductor_meta=None, custom_kernel: bool = False, filename: str | None = None, reset_to_zero_arg_names: list[str] | None = None, autotune_cache_info: dict[str, Any] | None = None) -> None: ...
    def is_statically_launchable(self):
        """
        Checks if every compiled kernel is statically launchable, which
        allows us to efficiently cache it in FXGraphCache
        """
    def recheck_autotune_cache(self, reload_kernel_from_src: Callable[[], CachingAutotuner]) -> None:
        """
        On cache load on static autotuner, we need to recheck the autotune cache, since
        a best config could have been found from a previous run
        """
    def set_compile_info(self, compile_id: CompileId | None, is_backward: bool) -> None: ...
    _reload_kernel: Incomplete
    def precompile(self, warm_cache_only: bool = False, reload_kernel: Callable[[], CachingAutotuner] | None = None, static_triton_bundle_key: str | None = None): ...
    def _precompile_worker(self) -> None: ...
    def _dynamic_scale_rblock(self) -> None: ...
    def _make_launchers(self) -> None: ...
    def prepare_for_pickle(self) -> tuple[Any, Any, Any, Any, Any]:
        """Drop stuff from triton.JITFunction that does not pickle.
        This must be called after precompile so that these things are no longer needed.
        Returns a tuple of old values
        """
    def prepare_for_caching(self) -> None:
        """
        Statically Launched CUDA Kernels have a raw cubin on them
        that we don't need to store in the cache(since TritonBundler handles the collection for us)
        """
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def get_device_interface(self): ...
    def _precompile_config(self, cfg: Config) -> CompileResult[_KernelType]:
        """Ahead of time compile a given autotuner config."""
    def _get_args_with_constexprs(self, args, launcher):
        """
        `args` is passed in with only the non-constexpr args (because the constexpr arg values
        depend on the config). However, in later triton versions, the constexpr args need to be
        added into the args list.
        """
    def bench(self, launcher, *args, with_profiler: bool = False, **kwargs):
        """Measure the performance of a given launcher"""
    def copy_args_to_cpu_if_needed(self, *args, **kwargs):
        """
        To support benchmarking in the presence of mutated args, we need to avoid
        autotuning contanminating them. We try to pass cloned args to the kernel.
        If those clones would increase the peak memory usage, however, we instead
        copy to cpu and restore them after each iteration. Figure out the args
        to be copied and do the copying.
        """
    def restore_args_from_cpu(self, cpu_copies) -> None: ...
    def reset_to_zero_args(self, *args, **kwargs) -> None: ...
    def maybe_clone_args(self, exclude: Container[str], *args, **kwargs) -> tuple[list[Any], dict[str, Any]]:
        """
        Prepare new args and kwargs by cloning any in-place buffers
        (that are not in the provided exclusion list), to avoid autotune
        contaminating them. Avoid cloning the other buffers because it
        leads to increased memory usage.
        """
    def clone_args(self, *args, **kwargs) -> tuple[list[Any], dict[str, Any]]: ...
    def benchmark_all_configs(self, *args, **kwargs): ...
    def autotune_to_one_config(self, *args, **kwargs) -> None:
        """Do the actual autotuning"""
    def save_gpu_kernel(self, stream, launcher) -> None: ...
    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate desecnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """
    def run(self, *args, stream, benchmark_run: bool = False, **kwargs): ...
    def _interpret_args_grid(self, args: tuple[Any, ...], cfg: Config) -> tuple[tuple[Any, ...], tuple[int, int, int]]: ...

class _ConstRepr:
    value: Incomplete
    def __init__(self, value: str) -> None: ...
    def __call__(self, _=None) -> str: ...

class CompileResult(Generic[_T]):
    kernel: Incomplete
    config: Incomplete
    compile_meta: Incomplete
    inductor_meta: Incomplete
    def __init__(self, kernel: _T, config: Config, compile_meta: dict[str, Any], inductor_meta: dict[str, Any]) -> None: ...
    def make_launcher(self) -> LauncherType: ...
    def _gen_launcher_code(self, scope, def_args, runner_args) -> LauncherType: ...
    def _get_arg_lists(self, arg_names, constexprs) -> tuple[list[str], list[str], OrderedSet[str]]:
        """
        Return a bunch of intermediate lists of args needed for generating
        launcher code.
        """

class CannotStaticallyLaunchKernel(Exception): ...

class StaticTritonCompileResult(CompileResult[StaticallyLaunchedCudaKernel]):
    """
    TritonCompileResult that uses StaticCudaLauncher,
    which vastly simplifies the setup and metadata needed to be kept.
    """
    @staticmethod
    def can_statically_launch(kernel: CompiledKernel, inductor_meta: dict[str, Any], triton_meta: dict[str, Any], heuristic_type: HeuristicType) -> StaticallyLaunchedCudaKernel | None: ...
    def reload_cubin_path(self) -> None:
        """
        When loading from cache on disk, we want to reload cubin
        files from their appropriate location on disc.
        """
    def make_launcher(self) -> LauncherType: ...

class TritonCompileResult(CompileResult[CompiledKernel]):
    """
    Upstream Triton CompileKernel can not be pickled.  This is a wrapper
    to support serialization and generate the launcher function.
    """
    @staticmethod
    def _kernel_metadata_cls(fields: tuple[str, ...]) -> Any: ...
    @staticmethod
    def _serialize_metadata(metadata):
        """
        Triton uses a nested class called KernelMetadata to store metadata information.
        Pickle does not work well with nested namedtuples, as the namedtuple doesn't appear
        in the toplevel namespace of the module. So these serialization/deser functions
        are used to convert the namedtuples to a dict and back.

        As for packed_metadata, depending on the triton backend, KernelMetadata can be
        a namedtuple, or a regular tuple! So the serialization function branches on whether
        the metadata to be serialized is a namedtuple or regular, serializable one.
        """
    @staticmethod
    def _deserialize_metadata(metadata): ...
    def __getstate__(self) -> dict[str, Any]: ...
    kernel: Incomplete
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def make_launcher(self) -> LauncherType:
        """
        Launching triton kernels is performance sensitive, we compile
        a custom Python function get the grid() and reorder the args to
        the underlying wrapper.
        """

def _find_names(obj): ...

collected_calls: list[Any]

def start_graph() -> None: ...
def end_graph(output_file): ...

class DebugAutotuner(CachingAutotuner):
    regex_filter: Incomplete
    with_profiler: Incomplete
    with_bandwidth_info: Incomplete
    cached: Incomplete
    def __init__(self, *args, regex_filter: str = '', with_profiler: bool = False, with_bandwidth_info: bool = True, **kwargs) -> None: ...
    precompile_time_taken_ns: Incomplete
    def run(self, *args, stream, **kwargs) -> None: ...

def hash_configs(configs: list[Config]):
    """
    Hash used to check for changes in configurations
    """
def cached_autotune(size_hints: list[int] | None, configs: list[Config], triton_meta, heuristic_type, filename=None, inductor_meta=None, custom_kernel: bool = False):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
def unique_configs(configs: list[Config]):
    """Remove duplicate configurations"""
def check_config(cfg, *, xnumel=None, ynumel=None, znumel=None) -> None: ...
def check_max_block(cfg: dict[str, int]):
    """
    Check that block sizes are within the maximum allowed.
    """
def _num_warps(num_warps, max_num_warps: int = 8, min_num_warps: int = 2, register_intensive: bool = False): ...
def _check_max_grid_x(size_hints, x, num_warps): ...
def triton_config(size_hints, x, y=None, z=None, num_stages: int = 1, num_elements_per_warp: int = 256, min_elem_per_thread: int = 0) -> Config:
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.

    num_elements_per_warp is a suggestion for controlling how many warps
    the triton config should contain. e.g.: if x=16, y=8, z=4 then
    num_elements = 16*8*4 = 512. Then if we set num_elements_per_warp=128,
    we'll launch 512 (elem) / 128 (elem/warp) = 4 warps. Note that it's
    just a suggestion, and sometimes other adjustment heuristics will
    override the num_elements_per_warp.

    min_elem_per_thread controls the minimum number of elements
    processed by each thread. It's always enforced.
    """
def _get_nd_reduction_numels(r: int, size_hints: dict[str, int]) -> dict[str, int]:
    """
    Converts a linear reduction numel to ND, in row major order.
    This order is often desirable as it presents opportunities to coalesce memory
    accesses.
    For example, if r = 64 and size_hints = [32,32], this function returns [32, 2].
    This unraveling works because both r and size_hints are powers of 2.
    """
def triton_config_reduction(size_hints, x: int, r: int, num_stages: int = 1, num_warps=None, register_intensive: bool = False) -> Config:
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
def _get_config(numels: dict[str, int]) -> dict[str, int]:
    '''
    Convert numels ("x", "r0_", etc.) to block sizes ("XBLOCK", "R0_BLOCK"), etc.
    '''
def triton_config_tiled_reduction(size_hints, x, y, r, num_stages: int = 1, register_intensive: bool = False):
    """
    Construct a tile reduction triton config with some adjustment
    heuristics based on size_hints. Size_hints is a tuple of numels in
    each tile dimension and will be rounded up to the nearest power of 2.
    """
def pointwise(size_hints, triton_meta, tile_hint=None, filename=None, min_elem_per_thread: int = 0, inductor_meta=None):
    """
    Construct @triton.heuristics() based on size_hints.
    """
def _reduction_configs(*, size_hints: dict[str, int], inductor_meta: dict[str, Any]) -> list[Config]: ...
def match_target_block_product(size_hints, tiling_scores, target_block_product, min_block_size: int = 1):
    """
    Distribute block sizes across dimensions according to tiling scores,
    aiming to match a target product of block sizes.
    """
def adapt_config_for_tiling(size_hints, tiling_scores, original_x, original_r, num_warps=None, num_stages: int = 1, register_intensive: bool = False, persistent_reduction: bool = False) -> Config:
    """
    Create an adapted configuration based on tiling scores,
    redistributing the same total block size (x * r) according to tiling scores.
    """
def reduction(size_hints, reduction_hint: bool = False, triton_meta=None, filename=None, inductor_meta=None):
    """args to @triton.heuristics()"""
def cooperative_reduction(size_hints, reduction_hint, triton_meta, filename, inductor_meta): ...
def _persistent_reduction_configs(size_hints, reduction_hint: bool = False, inductor_meta=None): ...
def persistent_reduction(size_hints, reduction_hint: bool = False, triton_meta=None, filename=None, inductor_meta=None): ...
def split_scan(size_hints, reduction_hint: bool = False, triton_meta=None, filename=None, inductor_meta=None):
    """Heuristic for TritonSplitScanKernel"""
def template(num_stages, num_warps, triton_meta, num_consumer_groups: int = 0, num_buffers_warp_spec: int = 0, filename=None, inductor_meta=None):
    """
    Compile a triton template
    """
def _pop_config_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract triton.Config options that should become kwargs"""
def config_to_dict(config: Config) -> dict[str, Any]: ...
def config_from_dict(config: dict[str, Any]) -> Config: ...
def fixed_config(config, filename, triton_meta, inductor_meta):
    """
    Used when the configuration is already decided at compile time
    """
def user_autotune(configs, triton_meta, filename=None, inductor_meta=None, custom_kernel: bool = False):
    """
    Compile a user defined triton kernel
    """
def foreach(triton_meta, num_warps, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """

@dataclasses.dataclass
class GridExpr:
    """Generate code for grid size expressions in launcher"""
    inductor_meta: dict[str, Any]
    mode: Literal['python', 'cpp'] = ...
    prefix: list[str] = dataclasses.field(default_factory=list)
    x_grid: str | int = ...
    y_grid: str | int = ...
    z_grid: str | int = ...
    def __post_init__(self) -> None: ...
    def generate(self, meta: dict[str, int]) -> None: ...
    def ceildiv(self, numel: str | int, block: None | int | str) -> str | int: ...
    def maximum(self, seq: list[int | str]) -> int | str:
        """Codegen for max function with constant folding, constants are represented as int"""
    def summation(self, seq: list[int | str]) -> int | str:
        """Codegen for sum function with constant folding, constants are represented as int"""
    def _constant_fold(self, fn: Callable[[list[int]], int], seq: list[int | str]) -> list[int | str]:
        """Constant fold through a commutative fn where ints are constants"""
    def assign_tmp(self, name: str, expr: str | int) -> str: ...
    @staticmethod
    def from_meta(inductor_meta: dict[str, Any], cfg: Config | dict[str, int], mode: Literal['python', 'cpp'] = 'python') -> GridExpr: ...
    def eval_slow(self, meta: dict[str, int]) -> tuple[int, int, int]: ...

class Grid1D(GridExpr):
    x_grid: Incomplete
    def generate(self, meta: dict[str, int]) -> None: ...

class Grid2D(GridExpr):
    x_grid: Incomplete
    y_grid: Incomplete
    def generate(self, meta: dict[str, int]) -> None: ...

class Grid3D(GridExpr):
    x_grid: Incomplete
    y_grid: Incomplete
    z_grid: Incomplete
    def generate(self, meta: dict[str, int]) -> None: ...

class Grid2DWithYZOverflow(GridExpr):
    x_grid: Incomplete
    y_grid: Incomplete
    z_grid: str
    def generate(self, meta: dict[str, int]) -> None: ...

class CooperativeReductionGrid(GridExpr):
    x_grid: Incomplete
    y_grid: Incomplete
    def generate(self, meta: dict[str, int]) -> None: ...

class SplitScanGrid(GridExpr):
    x_grid: Incomplete
    y_grid: str
    def generate(self, meta: dict[str, int]) -> None: ...

class FixedGrid(GridExpr):
    @staticmethod
    def setup_grid_as_args() -> dict[str, Any]:
        """Inductor meta so the launcher takes three extra grid arguments"""
    def generate(self, meta: dict[str, int]) -> None: ...

class PrecomputedGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None: ...

class ComboKernelGrid(GridExpr):
    x_grid: Incomplete
    y_grid: Incomplete
    z_grid: Incomplete
    def generate(self, meta: dict[str, int]): ...
    def combo_x_grid(self, xnumels: list[int | str], no_x_dims: list[bool], meta: dict[str, int]) -> str | int: ...

class SequentialComboKernelGrid(ComboKernelGrid):
    def combo_x_grid(self, xnumels: list[int | str], no_x_dims: list[bool], meta: dict[str, int]) -> str | int: ...

class RoundRobinComboKernelGrid(ComboKernelGrid):
    def combo_x_grid(self, xnumels: list[int | str], no_x_dims: list[bool], meta: dict[str, int]) -> str: ...
