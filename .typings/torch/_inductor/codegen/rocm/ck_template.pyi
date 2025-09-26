from .rocm_template import ArgInfo as ArgInfo
from _typeshed import Incomplete
from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate as ROCmTemplate
from torch._inductor.ir import IRNode as IRNode
from torch._inductor.utils import IndentedBuffer as IndentedBuffer
from typing import Any
from typing_extensions import override

class CKTemplate(ROCmTemplate):
    """
    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic
    """
    _TORCH_DTYPE_TO_CK: Incomplete
    def header(self) -> IndentedBuffer: ...
    def globals(self) -> IndentedBuffer: ...
    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str: ...
    @override
    def get_runtime_arg_info(self) -> list[ArgInfo]: ...
    @override
    def get_runtime_arg_values(self, **kwargs: Any) -> list[Any]:
        """
        Helper method to retrieve runtime args from generate kwargs
        """
