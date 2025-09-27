import dataclasses
from .runtime.runtime_utils import triton_cache_dir as triton_cache_dir
from .utils import GPU_KERNEL_BIN_EXTS as GPU_KERNEL_BIN_EXTS, _IS_WINDOWS as _IS_WINDOWS
from _typeshed import Incomplete
from torch._dynamo.utils import counters as counters, dynamo_timed as dynamo_timed, set_feature_use as set_feature_use
from torch._utils_internal import justknobs_check as justknobs_check
from torch.utils._filelock import FileLock as FileLock

log: Incomplete

@dataclasses.dataclass(frozen=True)
class TritonBundleEntry:
    """
    When we have compiled a triton kernel, we take note of that kernel by
    its triton generated hash, its device, and where this kernel is located.
    This is the minimum information we can use to later retrieve this kernel
    from file system.
    """
    kernel_hash: str
    device: int
    directory: str

@dataclasses.dataclass(frozen=True)
class TritonKernelArtifact:
    """
    Artifact for an individual kernel converted to bytes.
    Bytes could be a cubin, json, ttir, or ttgir.
    """
    filename: str
    payload: bytes = dataclasses.field(repr=False)

@dataclasses.dataclass(frozen=True)
class StaticallyLaunchedAutotuner:
    """
    Represents a statically compiled CachingAutotuner object that we can
    save directly in the cache. A CachingAutotuner is made up of a list of
    StaticTritonCompileResults, each of which uses the cubin from a TritonKernelArtifact.

    Statically saved here have their cubin files saved by a corresponding TritonBundleEntry.
    """
    cache_key: str
    kernel_name: str
    kernel: CachingAutotuner

@dataclasses.dataclass(frozen=True)
class TritonKernelArtifacts:
    """
    Collection of artifacts for a particular kernel.
    """
    kernel_hash: str
    device: int
    artifacts: list[TritonKernelArtifact]

@dataclasses.dataclass(frozen=True)
class TritonBundlerMetadata:
    """
    Metadata used for instrumentation
    """
    cached_kernel_names: list[str]
    statically_launched_kernel_names: list[str]

@dataclasses.dataclass(frozen=True)
class TritonBundle:
    """
    Serializable bundle to save into FXGraphCache
    """
    kernel_artifacts: list[TritonKernelArtifacts]
    static_autotuners: list[StaticallyLaunchedAutotuner]

class TritonBundler:
    """
    Lightweight Triton Kernel bundler that notes each time we compile a triton
    kernel. When collect is called, converts all the previously noted kernels and
    their artifacts into a structured bytes blob, and later when write is called
    it writes this structured blob back to file system.

    Intended Life cycle:
    - TritonBundler.begin_compile is called when we start compiling in Inductor
    - TritonBundler.put is called each time a Triton Kernel is compiled
    - TritonBundler.collect is called when a cache entry is being generated
    - TritonBundler.end_compile is called to indicate bundling is completed,
      collect will execute this function as well.
    - TritonBundler.read_and_emit is called when a cache entry is read
    """
    _entries: list[TritonBundleEntry] | None
    _static_autotuners: list[StaticallyLaunchedAutotuner] | None
    _REPLACE_BYTES: bytes
    @staticmethod
    def is_enabled() -> bool: ...
    @classmethod
    def begin_compile(cls) -> None:
        """
        Initializes the TritonBundler.
        The current TritonBundler bundle is finalized by TritonBundler.collect.
        """
    @classmethod
    def end_compile(cls) -> None:
        """
        Finalizes the TritonBundler. If collect is not yet called, it
        discards the current bundle.
        """
    @classmethod
    def put(cls, kernel_hash: str, device: int) -> None:
        """
        Lazily observes that we have seen a Triton kernel compilation. Remembers
        it for when collect is later called.
        """
    @classmethod
    def put_static_autotuner(cls, key: str, kernel: CachingAutotuner) -> None: ...
    @classmethod
    def collect_static_autotuners(cls) -> tuple[list[StaticallyLaunchedAutotuner], list[str]]: ...
    @classmethod
    def load_autotuners(cls, static_autotuners: list[StaticallyLaunchedAutotuner] | None) -> list[str]:
        """
        Load statically launchable CachingAutotuners into async_compile.CompiledTritonKernels
        cache.
        """
    @classmethod
    def collect(cls) -> tuple[TritonBundle, TritonBundlerMetadata | None]:
        """
        This is the main function called when a cache write happens. This function
        converts all the previously remembered kernels into bundled format so that
        it can be written into a cache entry.
        This function also finalizes the current bundle.
        """
    @staticmethod
    def read_and_emit(bundle: TritonBundle) -> TritonBundlerMetadata | None:
        """
        This is the main function called when a cache read happens. This function
        converts the bundled format back into individual files and writes them
        to the filesystem.

        NOTE: When we are writing to the filesystem, we assume exclusive access
        to the target directory.
        This means that if the target folder already exists and is non-empty,
        we bail out.
        Exclusive access means that no other process should be writing to
        or reading from the target directory.
        """
