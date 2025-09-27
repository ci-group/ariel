import sympy
from ..utils import sympy_product as sympy_product
from torch._inductor import config as config
from torch._inductor.codegen.simd import IterationRangesRoot as IterationRangesRoot, prefix_is_reduction as prefix_is_reduction
from torch._inductor.codegen.triton import TritonCSEVariable as TritonCSEVariable, TritonKernel as TritonKernel, triton_compute_type as triton_compute_type
from torch._inductor.runtime.triton_heuristics import SplitScanGrid as SplitScanGrid
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._sympy.functions import CeilDiv as CeilDiv

class TritonSplitScanKernel(TritonKernel):
    """Generates a triton kernel that supports ops.scan calls while also splitting
    the reduction dimension over multiple triton programs.

    For this kernel, loop numels will always take the form ``(xdim, rdim)``
    and the grid has the shape ``(CeilDiv(rdim, RBLOCK), xdim)``. Communication
    between blocks occurs within a global memory workspace buffer, which
    must be zero-filled before launching the kernel.

    Note that generation for ``ops.reduction`` is not supported.

    For details of the communication strategy, see
    https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    """
    no_x_dim: bool
    def __init__(self, tiling: dict[str, sympy.Expr], pid_cache=None, fixed_config=None, **kwargs) -> None: ...
    def should_use_persistent_reduction(self) -> bool: ...
    def should_use_cooperative_reduction(self) -> bool: ...
    def initialize_range_tree(self, pid_cache) -> None: ...
    def reduction(self, dtype, src_dtype, reduction_type, value) -> None: ...
    def scan(self, dtypes, combine_fn, values): ...
    def _get_heuristic(self): ...
    def _get_grid_type(self) -> type[SplitScanGrid]: ...
