import functools
from ...utils import IndentedBuffer as IndentedBuffer
from _typeshed import Incomplete
from dataclasses import dataclass
from torch._inductor import config as config
from torch._inductor.codegen.rocm.ck_tile_template import CKTileTemplate as CKTileTemplate
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel as ROCmTemplateKernel
from torch._inductor.codegen.rocm.rocm_template import ArgInfo as ArgInfo
from torch._inductor.ir import Buffer as Buffer, Layout as Layout
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any

log: Incomplete

def is_static_int(number): ...
def torch_layout_to_ck_layout(torch_layout): ...

@dataclass
class CKTileGemmOperation:
    layout_a: str
    layout_b: str
    layout_c: str
    datatype_a: str
    datatype_b: str
    datatype_c: str
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int
    m_is_padded: str
    n_is_padded: str
    k_is_padded: str
    pipeline: str
    scheduler: str
    epilogue: str
    def layout_repr(self): ...
    def dtype_repr(self): ...
    def tile_sizes(self): ...
    def name(self): ...
    def dict_items(self): ...

@functools.cache
def ops():
    """
    Generate the supported instance dataclasses
    """

class CKTileGemmTemplate(CKTileTemplate):
    """
    This class is used for rendering CK-Tile Universal GEMM kernels
    """
    gemm_template: str
    def __init__(self, input_nodes: list[Buffer], layout: Layout) -> None: ...
    def header(self) -> IndentedBuffer: ...
    def globals(self) -> IndentedBuffer: ...
    def check_dtypes(self, op: CKTileGemmOperation): ...
    def check_layouts(self, op: CKTileGemmOperation): ...
    def get_gemm_problem_size(self): ...
    def check_block_tiles(self, op: CKTileGemmOperation):
        """
        The contiguous dimension of a tensor must be divisible by the block tile size
        This helper function enforces it for the inputs and the output.
        """
    def check_alignments(self, op: CKTileGemmOperation):
        """
        The contiguous dimension of a tensor must be divisible by the vector load size.
        """
    def check_warp_tiles(self, op: CKTileGemmOperation): ...
    def check_block_tile_size(self, op: CKTileGemmOperation): ...
    def filter_op(self, op: CKTileGemmOperation):
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """
    def emit_ck_instance(self, op: CKTileGemmOperation):
        """
        This method is used to generate code which defines the type alias for the generated kernel class
        """
    output_node: Incomplete
    def render(self, kernel: ROCmTemplateKernel, op: CKTileGemmOperation, **kwargs) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        """
    def gen_ops(self):
        """
        Creates a list of `CKTileGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime.
        Such instances will be assigned +inf runtime by the autotune process.
        """
    @staticmethod
    def add_choices(choices, layout, input_nodes) -> None:
        """
        Add Composable Kernel Universal GEMM instance choices to the auto-tuning list.
        """
    def k_batch_choices(self, op: CKTileGemmOperation) -> tuple[int, ...]:
        """
        Returns a list of k_batch choices for the template.
        """
    def size_args(self):
        """
        Sizes and strides to be used for the kernel call
        """
    def get_runtime_arg_info(self) -> list[ArgInfo]: ...
    def get_runtime_arg_values(self, **kwargs: Any) -> list[Any]: ...
