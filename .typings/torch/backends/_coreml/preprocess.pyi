import torch
from _typeshed import Incomplete

CT_METADATA_VERSION: str
CT_METADATA_SOURCE: str

class ScalarType:
    Float: int
    Double: int
    Int: int
    Long: int
    Undefined: int

torch_to_mil_types: Incomplete

class CoreMLComputeUnit:
    CPU: str
    CPUAndGPU: str
    ALL: str

class CoreMLQuantizationMode:
    LINEAR: str
    LINEAR_SYMMETRIC: str
    NONE: str

def TensorSpec(shape, dtype=...): ...
def CompileSpec(inputs, outputs, backend=..., allow_low_precision: bool = True, quantization_mode=..., mlmodel_export_path=None, convert_to=None): ...
def _check_enumerated_shape(shape): ...
def _convert_to_mil_type(shape, dtype, name: str): ...
def preprocess(script_module: torch._C.ScriptObject, compile_spec: dict[str, tuple]): ...
