import dataclasses
from .. import config as config
from ..codecache import CudaKernelParamCache as CudaKernelParamCache
from ..ir import GraphPartitionSignature as GraphPartitionSignature, TMADescriptorExperimental as TMADescriptorExperimental, TMADescriptorStable as TMADescriptorStable, TensorBox as TensorBox
from ..utils import GPU_ALIGN_BYTES as GPU_ALIGN_BYTES, IndentedBuffer as IndentedBuffer, cache_on_self as cache_on_self, get_gpu_type as get_gpu_type
from ..virtualized import V as V
from .aoti_hipify_utils import maybe_hipify_code_wrapper as maybe_hipify_code_wrapper
from .common import TritonScratchWorkspace as TritonScratchWorkspace, get_device_op_overrides as get_device_op_overrides
from .cpp_utils import cexpr as cexpr
from .cpp_wrapper_cpu import CppWrapperCpu as CppWrapperCpu
from .multi_kernel import MultiKernelCall as MultiKernelCall
from .triton_utils import should_unwrap_unspec_arg as should_unwrap_unspec_arg
from .wrapper import PythonWrapperCodegen as PythonWrapperCodegen, SymbolicCallArg as SymbolicCallArg
from _typeshed import Incomplete
from torch import dtype as torch_dtype
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name as get_cpp_wrapper_cubin_path_name
from torch._inductor.runtime.runtime_utils import dynamo_timed as dynamo_timed
from typing import Any
from typing_extensions import Self

_cpp_string_literal_escapes: Incomplete
_cpp_string_literal_pattern: Incomplete

def cpp_string_literal(s: str) -> str: ...

@dataclasses.dataclass
class DeferredTritonCallWrapper:
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred generating the final wrapper around
    the triton kernel until right before the prefix is written.
    """
    wrapper_name: str
    kernel_name: str
    kernel_name_to_body: dict[str, str]
    arg_types: list[Any]
    def generate(self, wrapper: CppWrapperGpu):
        """
        Generate the GPU kernel definition, as well as load and launch code.
        """
    def generate_grid(self, prefix: IndentedBuffer, inductor_meta: dict[str, Any], params: dict[str, Any]): ...
    def generate_load_kernel(self, prefix, kernel_var_name, params) -> None: ...
    def generate_launch_kernel(self, prefix, wrapper, kernel_var_name, params) -> None: ...

class CppWrapperGpu(CppWrapperCpu):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """
    device: Incomplete
    device_codegen: Incomplete
    grid_id: Incomplete
    _kernel_name_to_body: dict[str, str]
    _triton_call_wrappers: dict[str, DeferredTritonCallWrapper]
    autotune_input_prefix: str
    def __init__(self) -> None: ...
    @staticmethod
    def create(is_subgraph: bool, subgraph_name: str | None, parent_wrapper: PythonWrapperCodegen | None, partition_signatures: GraphPartitionSignature | None = None): ...
    def write_header(self) -> None: ...
    @cache_on_self
    def write_tma_descriptor_helpers_once(self) -> None: ...
    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str: ...
    def get_autotuning_input_name(self, idx): ...
    def codegen_inputs(self): ...
    def _define_kernel_helper(self, kernel_name: str, kernel_body: str, metadata: str | None = None, gpu: bool = True, cpp_definition: str | None = None): ...
    def generate(self, is_inference): ...
    prefix: Incomplete
    def finalize_prefix(self) -> None:
        """Define the triton kernels now that autotuning is finished"""
    def generate_tma_descriptor(self, desc) -> None: ...
    def _generate_experimental_tma_descriptor(self, desc) -> None: ...
    def _generate_stable_tma_descriptor(self, desc) -> None: ...
    def generate_args_decl(self, code: IndentedBuffer | Self, call_args, arg_types, arg_signatures, is_triton_kernel: bool = True, workspace_size: int = 0):
        '''
        Generates any declarations of args to pass into a kernel call, and then returns the arg names.

        In more detail:
        * declarations: e.g. this function has a side effect of generating lines like `auto var_0 = ...;`
        * returns: a string with the list of args, e.g. "var_0, var_1"

        call_args: list of call arguments
        arg_types: list of argument types
        arg_signatures: list with signatures of all the args
        is_triton_kernel: whether these are passed into a triton kernel or not. In particular,
                          calls to triton kernels will have an additional global scratch space
                          arg injected at the front of the arg list.
        '''
    def _generate_kernel_call_helper(self, kernel_name: str, call_args, *, device=None, triton: bool = True, arg_types=None, raw_keys=None, raw_args=None, triton_meta=None, graph_name: str = '', original_fxnode_name=None):
        """
        Override the default value of argument 'gpu' to True here.
        generate_kernel_call can still be called with gpu=False because of
        a mix of cpu kernels and gpu kernels.
        """
    @staticmethod
    def prepare_triton_wrapper_args(call_args: list[Any], arg_types: list[Any]) -> tuple[list[Any], list[Any]]: ...
    def make_zero_buffer(self, name): ...

@dataclasses.dataclass
class UnwrapUnspecArg:
    """Marker that we need to call .item() on the tensor"""
    dtype: torch_dtype
