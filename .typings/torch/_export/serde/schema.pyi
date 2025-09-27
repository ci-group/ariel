from _typeshed import Incomplete
from dataclasses import dataclass, field
from enum import IntEnum
from torch._export.serde.union import _Union as _Union
from typing import Annotated

SCHEMA_VERSION: Incomplete
TREESPEC_VERSION: int

class ScalarType(IntEnum):
    UNKNOWN = 0
    BYTE = 1
    CHAR = 2
    SHORT = 3
    INT = 4
    LONG = 5
    HALF = 6
    FLOAT = 7
    DOUBLE = 8
    COMPLEXHALF = 9
    COMPLEXFLOAT = 10
    COMPLEXDOUBLE = 11
    BOOL = 12
    BFLOAT16 = 13
    UINT16 = 28
    FLOAT8E4M3FN = 29
    FLOAT8E5M2 = 30

class Layout(IntEnum):
    Unknown = 0
    SparseCoo = 1
    SparseCsr = 2
    SparseCsc = 3
    SparseBsr = 4
    SparseBsc = 5
    _mkldnn = 6
    Strided = 7

class MemoryFormat(IntEnum):
    Unknown = 0
    ContiguousFormat = 1
    ChannelsLast = 2
    ChannelsLast3d = 3
    PreserveFormat = 4

@dataclass
class Device:
    type: Annotated[str, 10]
    index: Annotated[int | None, 20] = ...

@dataclass(repr=False)
class SymExprHint(_Union):
    as_int: Annotated[int, 10]
    as_bool: Annotated[bool, 20]
    as_float: Annotated[float, 30]

@dataclass
class SymExpr:
    expr_str: Annotated[str, 10]
    hint: Annotated[SymExprHint | None, 20] = ...

@dataclass(repr=False)
class SymInt(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_int: Annotated[int, 20]

@dataclass(repr=False)
class SymFloat(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_float: Annotated[float, 20]

@dataclass(repr=False)
class SymBool(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_bool: Annotated[bool, 20]

@dataclass
class TensorMeta:
    dtype: Annotated[ScalarType, 10]
    sizes: Annotated[list[SymInt], 20]
    requires_grad: Annotated[bool, 30]
    device: Annotated[Device, 40]
    strides: Annotated[list[SymInt], 50]
    storage_offset: Annotated[SymInt, 60]
    layout: Annotated[Layout, 70]

@dataclass(repr=False)
class SymIntArgument(_Union):
    as_name: Annotated[str, 10]
    as_int: Annotated[int, 20]

@dataclass(repr=False)
class SymFloatArgument(_Union):
    as_name: Annotated[str, 10]
    as_float: Annotated[float, 20]

@dataclass(repr=False)
class SymBoolArgument(_Union):
    as_name: Annotated[str, 10]
    as_bool: Annotated[bool, 20]

@dataclass
class TensorArgument:
    name: Annotated[str, 10]

@dataclass
class TokenArgument:
    name: Annotated[str, 10]

@dataclass(repr=False)
class OptionalTensorArgument(_Union):
    as_tensor: Annotated[TensorArgument, 20]
    as_none: Annotated[bool, 10]

@dataclass
class GraphArgument:
    name: Annotated[str, 10]
    graph: Annotated['Graph', 20]

@dataclass
class CustomObjArgument:
    name: Annotated[str, 10]
    class_fqn: Annotated[str, 20]

@dataclass(repr=False)
class Argument(_Union):
    as_none: Annotated[bool, 10]
    as_tensor: Annotated[TensorArgument, 20]
    as_tensors: Annotated[list[TensorArgument], 30]
    as_int: Annotated[int, 50]
    as_ints: Annotated[list[int], 70]
    as_float: Annotated[float, 80]
    as_floats: Annotated[list[float], 90]
    as_string: Annotated[str, 100]
    as_strings: Annotated[list[str], 101]
    as_sym_int: Annotated[SymIntArgument, 110]
    as_sym_ints: Annotated[list[SymIntArgument], 120]
    as_scalar_type: Annotated[ScalarType, 130]
    as_memory_format: Annotated[MemoryFormat, 140]
    as_layout: Annotated[Layout, 150]
    as_device: Annotated[Device, 160]
    as_bool: Annotated[bool, 170]
    as_bools: Annotated[list[bool], 180]
    as_sym_bool: Annotated[SymBoolArgument, 182]
    as_sym_bools: Annotated[list[SymBoolArgument], 184]
    as_graph: Annotated[GraphArgument, 200]
    as_optional_tensors: Annotated[list[OptionalTensorArgument], 190]
    as_custom_obj: Annotated[CustomObjArgument, 210]
    as_operator: Annotated[str, 220]
    as_sym_float: Annotated[SymFloatArgument, 230]
    as_sym_floats: Annotated[list[SymFloatArgument], 240]
    as_optional_tensor: Annotated[OptionalTensorArgument, 250]

class ArgumentKind(IntEnum):
    UNKNOWN = 0
    POSITIONAL = 1
    KEYWORD = 2

@dataclass
class NamedArgument:
    name: Annotated[str, 10]
    arg: Annotated[Argument, 20]
    kind: Annotated[ArgumentKind | None, 30] = ...

@dataclass
class Node:
    target: Annotated[str, 10]
    inputs: Annotated[list[NamedArgument], 20]
    outputs: Annotated[list[Argument], 30]
    metadata: Annotated[dict[str, str], 40]
    is_hop_single_tensor_return: Annotated[bool | None, 50] = ...

@dataclass
class Graph:
    inputs: Annotated[list[Argument], 10]
    outputs: Annotated[list[Argument], 20]
    nodes: Annotated[list[Node], 30]
    tensor_values: Annotated[dict[str, TensorMeta], 40]
    sym_int_values: Annotated[dict[str, SymInt], 50]
    sym_bool_values: Annotated[dict[str, SymBool], 60]
    is_single_tensor_return: Annotated[bool, 70] = ...
    custom_obj_values: Annotated[dict[str, CustomObjArgument], 80] = field(default_factory=dict)
    sym_float_values: Annotated[dict[str, SymFloat], 90] = field(default_factory=dict)

@dataclass
class UserInputSpec:
    arg: Annotated[Argument, 10]

@dataclass(repr=False)
class ConstantValue(_Union):
    as_none: Annotated[bool, 10]
    as_int: Annotated[int, 20]
    as_float: Annotated[float, 30]
    as_string: Annotated[str, 40]
    as_bool: Annotated[bool, 50]

@dataclass
class InputToConstantInputSpec:
    name: Annotated[str, 10]
    value: Annotated[ConstantValue, 20]

@dataclass
class InputToParameterSpec:
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]

@dataclass
class InputToBufferSpec:
    arg: Annotated[TensorArgument, 10]
    buffer_name: Annotated[str, 20]
    persistent: Annotated[bool, 30]

@dataclass
class InputToTensorConstantSpec:
    arg: Annotated[TensorArgument, 10]
    tensor_constant_name: Annotated[str, 20]

@dataclass
class InputToCustomObjSpec:
    arg: Annotated[CustomObjArgument, 10]
    custom_obj_name: Annotated[str, 20]

@dataclass
class InputTokenSpec:
    arg: Annotated[TokenArgument, 10]

@dataclass(repr=False)
class InputSpec(_Union):
    user_input: Annotated[UserInputSpec, 10]
    parameter: Annotated[InputToParameterSpec, 20]
    buffer: Annotated[InputToBufferSpec, 30]
    tensor_constant: Annotated[InputToTensorConstantSpec, 40]
    custom_obj: Annotated[InputToCustomObjSpec, 50]
    token: Annotated[InputTokenSpec, 70]
    constant_input: Annotated[InputToConstantInputSpec, 60]

@dataclass
class UserOutputSpec:
    arg: Annotated[Argument, 10]

@dataclass
class LossOutputSpec:
    arg: Annotated[TensorArgument, 10]

@dataclass
class BufferMutationSpec:
    arg: Annotated[TensorArgument, 10]
    buffer_name: Annotated[str, 20]

@dataclass
class GradientToParameterSpec:
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]

@dataclass
class GradientToUserInputSpec:
    arg: Annotated[TensorArgument, 10]
    user_input_name: Annotated[str, 20]

@dataclass
class UserInputMutationSpec:
    arg: Annotated[TensorArgument, 10]
    user_input_name: Annotated[str, 20]

@dataclass
class OutputTokenSpec:
    arg: Annotated[TokenArgument, 10]

@dataclass(repr=False)
class OutputSpec(_Union):
    user_output: Annotated[UserOutputSpec, 10]
    loss_output: Annotated[LossOutputSpec, 20]
    buffer_mutation: Annotated[BufferMutationSpec, 30]
    gradient_to_parameter: Annotated[GradientToParameterSpec, 40]
    gradient_to_user_input: Annotated[GradientToUserInputSpec, 50]
    user_input_mutation: Annotated[UserInputMutationSpec, 60]
    token: Annotated[OutputTokenSpec, 70]

@dataclass
class GraphSignature:
    input_specs: Annotated[list[InputSpec], 10]
    output_specs: Annotated[list[OutputSpec], 20]

@dataclass
class RangeConstraint:
    min_val: Annotated[int | None, 10]
    max_val: Annotated[int | None, 20]

@dataclass
class ModuleCallSignature:
    inputs: Annotated[list[Argument], 10]
    outputs: Annotated[list[Argument], 20]
    in_spec: Annotated[str, 30]
    out_spec: Annotated[str, 40]
    forward_arg_names: Annotated[list[str] | None, 50] = ...

@dataclass
class ModuleCallEntry:
    fqn: Annotated[str, 10]
    signature: Annotated[ModuleCallSignature | None, 30] = ...

@dataclass
class NamedTupleDef:
    field_names: Annotated[list[str], 10]

@dataclass
class GraphModule:
    graph: Annotated[Graph, 10]
    signature: Annotated[GraphSignature, 50]
    module_call_graph: Annotated[list[ModuleCallEntry], 60]
    metadata: Annotated[dict[str, str], 40] = field(default_factory=dict)
    treespec_namedtuple_fields: Annotated[dict[str, NamedTupleDef], 70] = field(default_factory=dict)

@dataclass
class SchemaVersion:
    major: Annotated[int, 10]
    minor: Annotated[int, 20]

@dataclass
class ExportedProgram:
    graph_module: Annotated[GraphModule, 10]
    opset_version: Annotated[dict[str, int], 20]
    range_constraints: Annotated[dict[str, RangeConstraint], 30]
    schema_version: Annotated[SchemaVersion, 60]
    verifiers: Annotated[list[str], 70] = field(default_factory=list)
    torch_version: Annotated[str, 80] = ...

@dataclass
class Program:
    methods: Annotated[dict[str, ExportedProgram], 200]

@dataclass
class Model:
    name: Annotated[str, 10]
    tensorPaths: Annotated[dict[str, str], 20]
    program: Annotated[Program, 40]
    delegates: Annotated[dict[str, Program], 50]
    deviceAllocationMap: Annotated[dict[str, str], 60]
    constantPaths: Annotated[dict[str, str], 70]

@dataclass
class AOTInductorModelPickleData:
    library_basename: Annotated[str, 1]
    input_names: Annotated[list[str], 2]
    output_names: Annotated[list[str], 3]
    floating_point_input_dtype: Annotated[int | None, 4] = ...
    floating_point_output_dtype: Annotated[int | None, 5] = ...
    aot_inductor_model_is_cpu: Annotated[bool | None, 6] = ...

@dataclass
class ExternKernelNode:
    name: Annotated[str, 10]
    node: Annotated[Node, 20]

@dataclass
class ExternKernelNodes:
    nodes: Annotated[list[ExternKernelNode], 10]
