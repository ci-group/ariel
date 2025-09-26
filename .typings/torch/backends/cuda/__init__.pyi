import contextlib
import torch
from _typeshed import Incomplete
from torch._C import _SDPAParams as SDPAParams

__all__ = ['is_built', 'cuFFTPlanCacheAttrContextProp', 'cuFFTPlanCache', 'cuFFTPlanCacheManager', 'cuBLASModule', 'preferred_linalg_library', 'preferred_blas_library', 'preferred_rocm_fa_library', 'cufft_plan_cache', 'matmul', 'SDPAParams', 'enable_cudnn_sdp', 'cudnn_sdp_enabled', 'enable_flash_sdp', 'flash_sdp_enabled', 'enable_mem_efficient_sdp', 'mem_efficient_sdp_enabled', 'math_sdp_enabled', 'enable_math_sdp', 'allow_fp16_bf16_reduction_math_sdp', 'fp16_bf16_reduction_math_sdp_allowed', 'is_flash_attention_available', 'can_use_flash_attention', 'can_use_efficient_attention', 'can_use_cudnn_attention', 'sdp_kernel']

def is_built():
    """
    Return whether PyTorch is built with CUDA support.

    Note that this doesn't necessarily mean CUDA is available; just that if this PyTorch
    binary were run on a machine with working CUDA drivers and devices, we would be able to use it.
    """

class cuFFTPlanCacheAttrContextProp:
    getter: Incomplete
    setter: Incomplete
    def __init__(self, getter, setter) -> None: ...
    def __get__(self, obj, objtype): ...
    def __set__(self, obj, val) -> None: ...

class cuFFTPlanCache:
    """
    Represent a specific plan cache for a specific `device_index`.

    The attributes `size` and `max_size`, and method `clear`, can fetch and/ or
    change properties of the C++ cuFFT plan cache.
    """
    device_index: Incomplete
    def __init__(self, device_index) -> None: ...
    size: Incomplete
    max_size: Incomplete
    def clear(self): ...

class cuFFTPlanCacheManager:
    """
    Represent all cuFFT plan caches, return the cuFFTPlanCache for a given device when indexed.

    Finally, this object, when used directly as a `cuFFTPlanCache` object (e.g.,
    setting the `.max_size`) attribute, the current device's cuFFT plan cache is
    used.
    """
    __initialized: bool
    caches: Incomplete
    def __init__(self) -> None: ...
    def __getitem__(self, device): ...
    def __getattr__(self, name): ...
    def __setattr__(self, name, value): ...

class cuBLASModule:
    def __getattr__(self, name): ...
    def __setattr__(self, name, value): ...

def preferred_linalg_library(backend: None | str | torch._C._LinalgBackend = None) -> torch._C._LinalgBackend:
    '''
    Override the heuristic PyTorch uses to choose between cuSOLVER and MAGMA for CUDA linear algebra operations.

    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA linear algebra operation it often uses the cuSOLVER or MAGMA libraries,
    and if both are available it decides which to use with a heuristic.
    This flag (a :class:`str`) allows overriding those heuristics.

    * If `"cusolver"` is set then cuSOLVER will be used wherever possible.
    * If `"magma"` is set then MAGMA will be used wherever possible.
    * If `"default"` (the default) is set then heuristics will be used to pick between
      cuSOLVER and MAGMA if both are available.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_LINALG_PREFER_CUSOLVER=1 to set the preferred library to cuSOLVER
      globally.
      This flag only sets the initial value of the preferred library and the preferred library
      may still be overridden by this function call later in your script.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn\'t implement the operation(s) called.
    This flag may achieve better performance if PyTorch\'s heuristic library selection is incorrect
    for your application\'s inputs.

    Currently supported linalg operators:

    * :func:`torch.linalg.inv`
    * :func:`torch.linalg.inv_ex`
    * :func:`torch.linalg.cholesky`
    * :func:`torch.linalg.cholesky_ex`
    * :func:`torch.cholesky_solve`
    * :func:`torch.cholesky_inverse`
    * :func:`torch.linalg.lu_factor`
    * :func:`torch.linalg.lu`
    * :func:`torch.linalg.lu_solve`
    * :func:`torch.linalg.qr`
    * :func:`torch.linalg.eigh`
    * :func:`torch.linalg.eighvals`
    * :func:`torch.linalg.svd`
    * :func:`torch.linalg.svdvals`
    '''
def preferred_blas_library(backend: None | str | torch._C._BlasBackend = None) -> torch._C._BlasBackend:
    '''
    Override the library PyTorch uses for BLAS operations. Choose between cuBLAS, cuBLASLt, and CK [ROCm-only].

    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA BLAS operation it defaults to cuBLAS even if both cuBLAS and cuBLASLt are available.
    For PyTorch built for ROCm, hipBLAS, hipBLASLt, and CK may offer different performance.
    This flag (a :class:`str`) allows overriding which BLAS library to use.

    * If `"cublas"` is set then cuBLAS will be used wherever possible.
    * If `"cublaslt"` is set then cuBLASLt will be used wherever possible.
    * If `"ck"` is set then CK will be used wherever possible.
    * If `"default"` (the default) is set then heuristics will be used to pick between the other options.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_BLAS_PREFER_CUBLASLT=1 to set the preferred library to cuBLASLt
      globally.
      This flag only sets the initial value of the preferred library and the preferred library
      may still be overridden by this function call later in your script.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn\'t implement the operation(s) called.
    This flag may achieve better performance if PyTorch\'s library selection is incorrect
    for your application\'s inputs.

    '''
def preferred_rocm_fa_library(backend: None | str | torch._C._ROCmFABackend = None) -> torch._C._ROCmFABackend:
    '''
    [ROCm-only]
    Override the backend PyTorch uses in ROCm environments for Flash Attention. Choose between AOTriton and CK

    .. warning:: This flag is experimeental and subject to change.

    When Flash Attention is enabled and desired, PyTorch defaults to using AOTriton as the backend.
    This flag (a :class:`str`) allows users to override this backend to use composable_kernel

    * If `"default"` is set then the default backend will be used wherever possible. Currently AOTriton.
    * If `"aotriton"` is set then AOTriton will be used wherever possible.
    * If `"ck"` is set then CK will be used wherever possible.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_ROCM_FA_PREFER_CK=1 to set the preferred library to CK
      globally.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn\'t implement the operation(s) called.
    This flag may achieve better performance if PyTorch\'s library selection is incorrect
    for your application\'s inputs.
    '''
def flash_sdp_enabled():
    """
    .. warning:: This flag is beta and subject to change.

    Returns whether flash scaled dot product attention is enabled or not.
    """
def enable_flash_sdp(enabled: bool):
    """
    .. warning:: This flag is beta and subject to change.

    Enables or disables flash scaled dot product attention.
    """
def mem_efficient_sdp_enabled():
    """
    .. warning:: This flag is beta and subject to change.

    Returns whether memory efficient scaled dot product attention is enabled or not.
    """
def enable_mem_efficient_sdp(enabled: bool):
    """
    .. warning:: This flag is beta and subject to change.

    Enables or disables memory efficient scaled dot product attention.
    """
def math_sdp_enabled():
    """
    .. warning:: This flag is beta and subject to change.

    Returns whether math scaled dot product attention is enabled or not.
    """
def enable_math_sdp(enabled: bool):
    """
    .. warning:: This flag is beta and subject to change.

    Enables or disables math scaled dot product attention.
    """
def allow_fp16_bf16_reduction_math_sdp(enabled: bool):
    """
    .. warning:: This flag is beta and subject to change.

    Enables or disables fp16/bf16 reduction in math scaled dot product attention.
    """
def fp16_bf16_reduction_math_sdp_allowed():
    """
    .. warning:: This flag is beta and subject to change.

    Returns whether fp16/bf16 reduction in math scaled dot product attention is enabled or not.
    """
def is_flash_attention_available() -> bool:
    """Check if PyTorch was built with FlashAttention for scaled_dot_product_attention.

    Returns:
        True if FlashAttention is built and available; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
def can_use_flash_attention(params: SDPAParams, debug: bool = False) -> bool:
    """Check if FlashAttention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why FlashAttention could not be run.
            Defaults to False.

    Returns:
        True if FlashAttention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
def can_use_efficient_attention(params: SDPAParams, debug: bool = False) -> bool:
    """Check if efficient_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why efficient_attention could not be run.
            Defaults to False.

    Returns:
        True if efficient_attention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
def can_use_cudnn_attention(params: SDPAParams, debug: bool = False) -> bool:
    """Check if cudnn_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why cuDNN attention could not be run.
            Defaults to False.

    Returns:
        True if cuDNN can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
def cudnn_sdp_enabled():
    """
    .. warning:: This flag is beta and subject to change.

    Returns whether cuDNN scaled dot product attention is enabled or not.
    """
def enable_cudnn_sdp(enabled: bool):
    """
    .. warning:: This flag is beta and subject to change.

    Enables or disables cuDNN scaled dot product attention.
    """
@contextlib.contextmanager
def sdp_kernel(enable_flash: bool = True, enable_math: bool = True, enable_mem_efficient: bool = True, enable_cudnn: bool = True):
    """
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """

cufft_plan_cache: Incomplete
matmul: Incomplete
