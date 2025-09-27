from cutlass_library.gemm_operation import *
from cutlass_library.library import *
from ..cutlass_utils import try_import_cutlass as try_import_cutlass
from _typeshed import Incomplete

_LOGGER: Incomplete

class EmitGemmUniversal3xInstanceWithEVT:
    """Responsible for emitting a CUTLASS 3.x template definition"""
    operation_suffix: Incomplete
    includes: Incomplete
    builtin_epilogue_functor_template: str
    evt_name: Incomplete
    gemm_template: str
    def __init__(self, operation_suffix: str = '', evt_name=None) -> None: ...
    def instance_template(self): ...
    def emit_block_scale_epilogue_functor(self, operation): ...
    @staticmethod
    def pointerize_if_grouped(operation, layout): ...
    @staticmethod
    def problem_shape(operation): ...
    def emit(self, operation):
        """Given a gem operation, emits a template definition of the operation"""
