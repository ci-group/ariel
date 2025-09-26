import atexit
import functools
import torch
import types
from ... import config as config
from ...ir import Layout as Layout
from ...runtime.runtime_utils import cache_dir as cache_dir
from ...virtualized import V as V
from ..cpp_utils import DTYPE_TO_CPP as DTYPE_TO_CPP
from .cuda_env import get_cuda_arch as get_cuda_arch, get_cuda_version as get_cuda_version
from _typeshed import Incomplete
from dataclasses import dataclass
from pathlib import Path
from torch._inductor.utils import clear_on_fresh_cache as clear_on_fresh_cache
from typing import Any

log: Incomplete
CUTLASS_OPERATION_KIND: str

@atexit.register
def move_cutlass_compiled_cache() -> None:
    """Move CUTLASS compiled cache file to the cache directory if it exists."""
def _rename_cutlass_import(content: str, cutlass_modules: list[str]) -> str: ...
@functools.cache
def try_import_cutlass() -> bool:
    """
    We want to support three ways of passing in CUTLASS:
    1. fbcode, handled by the internal build system.
    2. pip install nvidia-cutlass, which provides the cutlass_library package
       and the header files in the cutlass_library/source directory.
    3. User specifies cutlass_dir. The default is ../third_party/cutlass/,
       which is the directory when developers build from source.
    """
def _normalize_cuda_arch(arch: str) -> str: ...

@dataclass
class CUTLASSArgs:
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """
    architectures: str | None = ...
    cuda_version: str | None = ...
    instantiation_level: str | None = ...
    operations: str | None = ...
    build_dir = ...
    curr_build_dir = ...
    generator_target = ...
    kernels = ...
    ignore_kernels = ...
    exclude_kernels = ...
    kernel_filter_file: None = ...
    selected_kernel_list: None = ...
    interface_dir: None = ...
    filter_by_cc = ...
    disable_full_archs_compilation = ...
    def __post_init__(self) -> None: ...

@clear_on_fresh_cache
@functools.cache
def _gen_ops_cached(arch, version) -> dict[Any, Any]: ...
def gen_ops() -> dict[Any, Any]:
    """
    Generates all supported CUTLASS operations.
    """

DTYPE_TO_CUTLASS_TYPE: Incomplete

def torch_dtype_to_cutlass_type(torch_dtype: torch.dtype) -> cutlass_library.library.DataType: ...
def dtype_match(torch_dtype: torch.dtype | None, cutlass_dtype: cutlass_library.library.DataType) -> bool: ...
def get_accumulator_dtype(input_torch_dtypes: list[torch.dtype]) -> torch.dtype | None:
    """
    Given a pair of input torch dtypes, returns the inferred accumulator torch dtype.
    """
def get_alignments(torch_dtype: torch.dtype) -> list[int]:
    """
    Returns all possible valid CUTLASS alignments in terms of the number of elements for a given dtype.
    CUTLASS gemm / conv SM80 APIs support 16 bytes max alignment, and 2 bytes min alignment.
    """
def get_max_alignment(inductor_layout: Layout) -> int:
    """
    Returns the max alignment (in terms of number of elements) for a given Inductor Layout.
    """

class CUDACompileSourceCapturingContext:
    sources: Incomplete
    _compile_patch: Incomplete
    def __init__(self) -> None: ...
    def __enter__(self, *args, **kwargs): ...
    def __exit__(self, *args, **kwargs) -> None: ...

def cuda_standalone_runner_compile_command(srcpath: Path, exepath: Path): ...
