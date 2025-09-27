from torch.utils._triton import has_triton as has_triton
from triton.language import core

def enable_triton(lib_dir: str | None = None) -> dict[str, str]:
    """
    Enable NVSHMEM device functions for Triton. It performs a NVSHMEM
    device-side initialization on the kernel module created by Triton.

    Args:
        lib_dir (Optional[str]): The directory where the NVSHMEM device library
        is located. If not provided, it will use the default path where NVSHMEM
        wheel is installed.

    Returns:
        dict[str, str]: A dictionary containing the NVSHMEM device library name
        and path.
    """
@core.extern
def putmem_block(dst, src, nelems, pe, _builder=None): ...
@core.extern
def getmem_block(dst, src, nelems, pe, _builder=None): ...
@core.extern
def putmem_signal_block(dst, src, nelems, sig_addr, signal, sig_op, pe, _builder=None): ...
@core.extern
def wait_until(ivar, cmp, cmp_val, _builder=None): ...
@core.extern
def signal_wait_until(sig_addr, cmp, cmp_val, _builder=None): ...
@core.extern
def fence(_builder=None): ...
@core.extern
def quiet(_builder=None): ...
