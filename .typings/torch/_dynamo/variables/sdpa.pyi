from ..bytecode_transformation import create_call_function as create_call_function
from ..exc import Unsupported as Unsupported
from ..source import AttrSource as AttrSource
from .base import VariableTracker as VariableTracker
from _typeshed import Incomplete
from torch._dynamo.codegen import PyCodegen as PyCodegen
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator

PARAM_NAMES: Incomplete

class SDPAParamsVariable(VariableTracker):
    """Represents the c++ params struct for scaled dot product attention.
    This is a read-only container."""
    @staticmethod
    def create(tx: InstructionTranslator, value, source): ...
    proxy: Incomplete
    param_vars: Incomplete
    def __init__(self, proxy, param_vars, **kwargs) -> None: ...
    def reconstruct(self, codegen: PyCodegen): ...
    def as_proxy(self): ...
    def var_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...
    @staticmethod
    def is_sdpa_params(value): ...
