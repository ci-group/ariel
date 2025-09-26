from _typeshed import Incomplete
from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate as ROCmTemplate
from torch._inductor.ir import IRNode as IRNode
from torch._inductor.utils import IndentedBuffer as IndentedBuffer

class CKTileTemplate(ROCmTemplate):
    """
    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic
    """
    _TORCH_DTYPE_TO_CK: Incomplete
    ck_dtype_to_size: Incomplete
    def header(self) -> IndentedBuffer: ...
    def globals(self) -> IndentedBuffer: ...
    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str: ...
