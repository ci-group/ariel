import functools
from .triton_compat import ASTSource as ASTSource, CompiledKernel as CompiledKernel
from _typeshed import Incomplete
from typing import Any
from typing_extensions import Unpack

class StaticallyLaunchedCudaKernel:
    """
    Parses the metadata of a CompiledKernel from Triton into a structure that can
    launch the cuda kernel directly. Only works for triton kernels compiled to cubin.

    Doing this avoids C++ codegen and compilation during compile, since we can use a
    statically compiled library to launch the kernel. To avoid mallocing for the arguments,
    we have a launcher for different numbers of arguments up to a max. StaticCudaLauncher
    only supports # of arguments up until 10 for now.

    Workflow:
    Compile time:
    1. Compile a kernel with triton and get a CompiledKernel
    2. Instantiate kernel = StaticallyLaunchedCudaKernel(triton_kernel)
    3. Write to a cubin file: kernel.write_cubin_to_file(filepath)
    4. Call kernel.load_kernel() (CUDA should be initialized by this point) to load the cubin
    Runtime:
    5. Call kernel.run(grid, stream, args) to launch the kernel

    Note that after step 3, StaticallyLaunchedCudaKernel is fully pickleable/serializable.
    This allows it to be cached by FXGraphCache/TritonBundler, as well as sent from the worker
    to the parent process in inductor.

    There are two main versions of triton that we wish to support: 3.3 and 3.2. Triton makes considerable changes
    to how it handles constants in 3.3, so there's some special logic necessary to handle both versions.
    """
    name: Incomplete
    cubin_raw: Incomplete
    cubin_path: Incomplete
    arg_names: Incomplete
    declared_constexprs: Incomplete
    hash: Incomplete
    num_warps: Incomplete
    shared: Incomplete
    has_global_scratch: bool
    arg_tys: Incomplete
    function: int | None
    def __init__(self, kernel: CompiledKernel) -> None: ...
    def reload_cubin_from_raw(self, filepath: str) -> str:
        """
        If the cubin file triton generated gets deleted under us, we can
        reload it from the raw cubin file.
        """
    def load_kernel(self, device: int) -> None: ...
    @staticmethod
    @functools.lru_cache
    def type_mappings() -> dict[str, str]: ...
    def extract_type(self, ty: str) -> str:
        """
        Takes a triton type from CompiledKernel.signature and
        converts it into a single char encoding. _StaticCudaLauncher
        will switch on this char to figure out what type the underlying
        value should be passed to the triton kernel as.
        """
    full_constexprs: Incomplete
    def arg_ty_from_signature(self, src: ASTSource) -> str: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def run(self, grid_x: int, grid_y: int, grid_z: int, stream: int, *args: Unpack[tuple[object, ...]]) -> None:
        """Actually run the kernel at runtime. This function is the hot codepath."""
