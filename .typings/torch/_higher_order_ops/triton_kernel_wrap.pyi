import dataclasses
import sympy
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import SymInt as SymInt, Tensor as Tensor
from torch._C import DispatchKey as DispatchKey
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator
from torch._dynamo.variables.constant import ConstantVariable as ConstantVariable
from torch._dynamo.variables.functions import TritonKernelVariable as TritonKernelVariable
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._prims_common import clone_preserve_strides as clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch._subclasses.functional_tensor import BaseFunctionalizeAPI as BaseFunctionalizeAPI
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, track_tensor_tree as track_tensor_tree
from torch.fx.experimental.symbolic_shapes import guard_scalar as guard_scalar
from torch.fx.proxy import Proxy as Proxy
from torch.types import IntLikeType as IntLikeType
from torch.utils._triton import has_triton as has_triton
from triton._C.libtriton.ir import module as TritonIRModule
from triton.runtime.autotuner import Autotuner, Config as TritonConfig
from triton.runtime.jit import JITFunction
from typing import Any, Callable
from typing_extensions import Never

TritonMetaParamsType = dict[str, int]
TritonGridTupleType: Incomplete
TritonGridCallableType = Callable[[TritonMetaParamsType], tuple[int, ...]]
TritonGridType = TritonGridTupleType | TritonGridCallableType

class Autotuner: ...
class JITFunction: ...
TritonKernelType = Autotuner | JITFunction
TritonAutotunerType = Autotuner
log: Incomplete
TMAExperimentalMetadata = tuple[str, tuple[list[IntLikeType], list[IntLikeType], IntLikeType]]
TMAStableMetadata = tuple[str, tuple[list[IntLikeType]]]

def create_tma_experimental_metadata(dims: list[IntLikeType], block_dims: list[IntLikeType], element_size: IntLikeType) -> TMAExperimentalMetadata: ...
def maybe_unpack_tma_experimental_metadata(tma_meta: TMAExperimentalMetadata | TMAStableMetadata) -> tuple[list[IntLikeType], list[IntLikeType], IntLikeType] | None: ...
def create_tma_stable_metadata(block_shape: list[IntLikeType]) -> TMAStableMetadata: ...
def maybe_unpack_tma_stable_metadata(tma_meta: TMAExperimentalMetadata | TMAStableMetadata) -> tuple[list[IntLikeType]] | None: ...
TMADescriptorMetadata = dict[str, TMAExperimentalMetadata | TMAStableMetadata]

class KernelSideTable:
    id_to_kernel: dict[int, 'TritonKernelType']
    kernel_to_id: dict['TritonKernelType', int]
    constant_args: dict[int, dict[str, Any]]
    lock: Incomplete
    def add_kernel(self, kernel: TritonKernelType) -> int: ...
    def get_kernel(self, idx: int) -> TritonKernelType: ...
    def add_constant_args(self, args: dict[str, Any]) -> int: ...
    def get_constant_args(self, idx: int) -> dict[str, Any]: ...
    def reset_table(self) -> None: ...

kernel_side_table: Incomplete

@dataclasses.dataclass(frozen=True)
class Param:
    idx: int

@dataclasses.dataclass(frozen=True)
class Intermediate:
    idx: int
    def fake(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class Op:
    name: str
    fn_call_name: str | None
    args: list[Param | Intermediate]
    ret: Intermediate = dataclasses.field(repr=False)
    sub_idx: int | None = ...
    is_pure: bool = ...
    def __post_init__(self) -> None: ...

def generate_ttir(kernel: TritonKernelType, kwargs: dict[str, Any], tma_descriptor_metadata: TMADescriptorMetadata) -> tuple['TritonIRModule', list[str]]:
    """
    Uses Triton's internal code generation to create TTIR
    """
def ttir_to_functions(ttir_module: TritonIRModule) -> dict[str, dict[Intermediate, list[Op]]]:
    """
    Walk the `ttir_module` bottom up to mine the `functions` from
    the structured MLIR entities representing the Triton kernel
    (mlir::Operation, mlir::Block, mlir::Region).
    """

class MemoizeWithCycleCheck:
    fn: Callable[..., Any]
    cache: dict[tuple[Any], Any]
    def __init__(self, fn: Callable[..., Any]) -> None: ...
    def __call__(self, functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str, *args: Any) -> list[bool]: ...
    def reset(self) -> None: ...

@MemoizeWithCycleCheck
def get_tma_stores(functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str) -> set[Intermediate | Param]:
    """
    Identifies all intermediates and parameters that are written to by a
    `tt.experimental_descriptor_store`. It tracks only the specific values
    written to via experimental_descriptor_store and the input values to
    `tt.reinterpret_tensor_descriptor` used to construct the direct inputs
    to tt.experimental_descriptor_store - not any recursive values
    used to construct those values.

    For example: for
      tt.reinterpret_tensor_descriptor(Intermediate(idx=0), ...)
      Intermediate(idx=1) = tt.experimental_descriptor_store(Intermediate(idx=0), ...)
    this function will return [Intermediate(idx=0), Intermediate(idx=1)],

    However
      Intermediate(idx=4) = arith.addptr(Intermediate(idx=2), Intermediate(idx=3))
      Intermediate(idx=5) = tt.experimental_descriptor_store(Intermediate(idx=4), ...)
      tt.experimental_descriptor_store(Intermediate(idx=5), ...)
    this function will mark only idx=4 and idx=5 (but not idx=2 or idx=3)

    If an intermediate/parameter is passed into a function and is written to
    via experimental_descriptor_store within that function, the argument to the
    function will also be marked.
    """
@MemoizeWithCycleCheck
def analyze_kernel_mutations(functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str, num_args: int) -> list[bool]:
    """
    Analyzes the graph to detect all sinks from a predefined list of sinks
    by using triton's MemWrite trait list. NOTE: What if triton exposed this?
    From each sink, it traverses the CFG backwards to identify all the input
    pointers that are mutated.
    """
def identify_mutated_tensors(kernel: TritonKernelType, kwargs: dict[str, Any], tma_descriptor_metadata: TMADescriptorMetadata) -> list[str]:
    """
    Given a triton kernel and the arguments for this kernel, this function
    1) Retrieves the TTIR converted version of the kernel from Triton's API.
    2) Parses the TTIR and creates a control flow graph
    3) Analyzes the graph to detect all input tensor mutations
    """

class TritonKernelWrapperMutation(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any]) -> Any: ...

triton_kernel_wrapper_mutation: Incomplete

class TritonKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any], tensors_to_clone: list[str]) -> dict[str, Any]: ...

triton_kernel_wrapper_functional: Incomplete

def triton_kernel_wrapper_mutation_dense(*, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any]) -> None: ...
def triton_kernel_wrapper_mutation_fake_tensor_mode(mode: FakeTensorMode, *, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any]) -> None: ...
def _(*, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any]) -> None: ...
def trace_triton_kernel_wrapper(proxy_mode: ProxyTorchDispatchMode, func_overload: Callable[..., Any], node_args: dict[str, Any]) -> dict[str, Any] | None: ...
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(mode: ProxyTorchDispatchMode, *, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any]) -> None: ...
def get_mutated_tensors(kernel_idx: int, constant_args_idx: int, kwargs: dict[str, Any], tma_descriptor_metadata: TMADescriptorMetadata) -> list[str]: ...
@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(ctx: BaseFunctionalizeAPI, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any]) -> None: ...
def triton_kernel_wrapper_functional_dense(*, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any], tensors_to_clone: list[str]) -> dict[str, Any]: ...
def triton_kernel_wrapper_functional_fake_tensor_mode(mode: FakeTensorMode, *, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any], tensors_to_clone: list[str]) -> dict[str, Any]: ...
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(mode: ProxyTorchDispatchMode, *, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any], tensors_to_clone: list[str]) -> dict[str, Any]: ...
@triton_kernel_wrapper_functional.py_functionalize_impl
def triton_kernel_wrapper_functional_functionalize(ctx: BaseFunctionalizeAPI, kernel_idx: int, constant_args_idx: int, grid: list['TritonGridType'], tma_descriptor_metadata: TMADescriptorMetadata, kwargs: dict[str, Any], tensors_to_clone: list[str]) -> dict[str, Any]: ...

class TritonHOPifier:
    """Orchestrator for converting a user-defined triton kernel into a call
    to the triton_kernel_wrapper_mutation HOP.

    It has two main use cases.

    1. When Dynamo sees a triton kernel, it wraps it into a TritonKernelVariable
    and uses the TritonHOPifier to convert calls to the TritonKernelVariable
    into a call to the HOP.

    2. In order to capture a user-defined triton kernel while performing
    tracing (via make_fx or non-strict export), a user must annotate their
    triton kernel with the `wrap_triton` decorator. The decorator uses
    TritonHOPifier to convert calls to the triton kernel into a call
    to the HOP (which can then be traced).

    Because Dynamo has its own calling conventions for e.g. invoking a user-defined function
    TritonHOPifier is an abstract class that can be overridden by its subclasses.
    """
    def raise_unsupported(self, msg: str) -> Never: ...
    def is_callable(self, maybe_callable: Any) -> bool: ...
    def get_value(self, val: Any) -> Any: ...
    def call_grid(self, grid, meta, tx) -> tuple[int | sympy.Expr | SymInt, ...] | tuple['Proxy', ...]: ...
    def wrap_user_defined_obj(self, user_obj: Any, tx: InstructionTranslator | None, variable: TritonKernelVariable | TraceableTritonKernelWrapper | None, name: str) -> Any: ...
    def call_user_defined_fn(self, user_fn: Callable[..., Any], args: list, kwargs: dict, tx: InstructionTranslator | None, variable: TritonKernelVariable | TraceableTritonKernelWrapper | None) -> Any: ...
    def maybe_unpack_configs(self, configs: list['TritonConfig'], tx: InstructionTranslator | None) -> list['TritonConfig']: ...
    def maybe_unpack_heuristic_result(self, result: Any) -> Any: ...
    @staticmethod
    def do_prune_configs(autotuner: TritonAutotunerType, early_config_prune: Callable | None, perf_model: Callable | None, top_k: float, configs: list, named_args: dict, kwargs: dict) -> list['TritonConfig']: ...
    def call_HOP(self, variable, grids, combined_args: dict[str, Any], tx) -> ConstantVariable | None: ...
    def check_grid(self, grid) -> tuple[int | sympy.Expr | SymInt, ...] | tuple['Proxy', ...]: ...
    def init_variable(self, variable: TraceableTritonKernelWrapper | TritonKernelVariable, kernel: TritonKernelType, kernel_idx: int | None, grid: TritonGridType | None) -> None: ...
    def call_getitem(self, variable: TritonKernelVariable | TraceableTritonKernelWrapper, args: Sequence[Any]) -> TritonKernelVariable | TraceableTritonKernelWrapper: ...
    def call_run(self, variable: TritonKernelVariable | TraceableTritonKernelWrapper, args: Sequence[Any], kwargs: dict[str, Any], tx: InstructionTranslator | None) -> ConstantVariable | None: ...
    def call_triton_kernel(self, variable: TritonKernelVariable | TraceableTritonKernelWrapper, args: Sequence[Any], kwargs: dict[str, Any], tx: InstructionTranslator | None) -> ConstantVariable | None: ...

class TracingTritonHOPifier(TritonHOPifier):
    def raise_unsupported(self, msg: str) -> Never: ...
    def is_callable(self, maybe_callable: Any) -> bool: ...
    def get_value(self, val: Any) -> Any: ...
    def call_grid(self, grid: TritonGridCallableType, meta: TritonMetaParamsType, tx: None) -> tuple[int | sympy.Expr | SymInt, ...]: ...
    def wrap_user_defined_obj(self, user_obj: Any, tx: InstructionTranslator | None, variable: TritonKernelVariable | TraceableTritonKernelWrapper | None, name: str) -> Any: ...
    def call_user_defined_fn(self, user_fn: Callable[..., Any], args: list, kwargs: dict, tx: InstructionTranslator | None, variable: TritonKernelVariable | TraceableTritonKernelWrapper | None) -> Any: ...
    def maybe_unpack_configs(self, configs: list['TritonConfig'], tx: InstructionTranslator | None) -> list['TritonConfig']: ...
    def maybe_unpack_heuristic_result(self, result: Any) -> Any: ...
    def check_grid(self, grid: TritonGridType) -> tuple[int | sympy.Expr | SymInt, ...]: ...
    def store_non_graphable_args(self, combined_args: dict[str, Any]) -> tuple[dict, int]:
        """
        Some args cannot be stored in the FX graph.
        Put them in the side table.
        """
    def call_HOP(self, variable: TraceableTritonKernelWrapper, grids: list['TritonGridTupleType'], combined_args: dict[str, Any], tx: None) -> None: ...

tracing_triton_hopifier_singleton: Incomplete

class TraceableTritonKernelWrapper:
    kernel: TritonKernelType
    kernel_idx: int | None
    grid: TritonGridType | None
    def __init__(self, kernel: TritonKernelType, kernel_idx: int | None, grid: TritonGridType | None) -> None: ...
    def __getitem__(self, *args: Sequence[Any]) -> TraceableTritonKernelWrapper: ...
    def run(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any: ...
    def __call__(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any: ...
    def specialize_symbolic(self, arg: Sequence[Any]) -> Any: ...
