import ctypes
from _typeshed import Incomplete
from typing import Any

def _get_cuda_library() -> ctypes.CDLL: ...
def _check_cuda(result: int) -> None: ...
def _get_nvrtc_library() -> ctypes.CDLL: ...
def _nvrtc_compile(kernel_source: str, kernel_name: str, compute_capability: str | None = None, header_code: str = '', cuda_include_dirs: list | None = None, nvcc_options: list | None = None) -> bytes:
    '''
    Compiles a CUDA kernel using NVRTC and returns the PTX code.

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, None): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        header_code (str, optional): Additional header code to prepend to the kernel source
        cuda_include_dirs (list, None): List of directories containing CUDA headers
        nvcc_options (list, None): Additional options to pass to NVRTC

    Returns:
        str: The compiled PTX code
    '''

class _CudaModule:
    _module: Incomplete
    _kernels: dict[str, _CudaKernel]
    def __init__(self, module: ctypes.c_void_p) -> None: ...
    def __getattr__(self, name: str) -> _CudaKernel: ...

class _CudaKernel:
    """
    Represents a compiled CUDA kernel that can be called with PyTorch tensors.
    """
    func: Incomplete
    module: Incomplete
    def __init__(self, func: ctypes.c_void_p, module: ctypes.c_void_p) -> None: ...
    def __call__(self, grid: tuple[int, int, int] = (1, 1, 1), block: tuple[int, int, int] = (1, 1, 1), args: list | None = None, shared_mem: int = 0, stream: Any | None = None) -> None:
        """
        Call the compiled CUDA kernel

        Args:
            grid (tuple): Grid dimensions (grid_x, grid_y, grid_z)
            block (tuple): Block dimensions (block_x, block_y, block_z)
            args (list): List of arguments to pass to the kernel.
                         PyTorch tensor arguments will be automatically converted to pointers.
            shared_mem (int): Shared memory size in bytes
            stream (torch.cuda.Stream): CUDA stream to use. If None, uses current stream.
        """

def _cuda_load_module(ptx: str | bytes, kernel_names: list[str] | None = None) -> _CudaModule | dict[str, '_CudaKernel']:
    """
    Loads a CUDA module from PTX code and returns a module object that can access kernels.

    Args:
        ptx (bytes or str): The PTX code to load
        kernel_names (list, optional): List of kernel names to extract from the module.
                                      If None, will return a module object with __getattr__.

    Returns:
        object: If kernel_names is None, returns a module object with __getattr__ to access kernels.
               If kernel_names is provided, returns a dict mapping kernel names to _CudaKernel objects.
    """
def _get_device_index(device: Any, optional: bool = False, allow_cpu: bool = False) -> int:
    """Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
