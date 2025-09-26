from ..ir import GraphPartitionSignature as GraphPartitionSignature
from ..virtualized import V as V
from .cpp_wrapper_gpu import CppWrapperGpu as CppWrapperGpu
from .wrapper import PythonWrapperCodegen as PythonWrapperCodegen
from typing import Any

class CppWrapperMps(CppWrapperGpu):
    @staticmethod
    def create(is_subgraph: bool, subgraph_name: str | None, parent_wrapper: PythonWrapperCodegen | None, partition_signatures: GraphPartitionSignature | None = None) -> CppWrapperMps: ...
    def _generate_kernel_call_helper(self, kernel_name: str, call_args: list[str], arg_types: list[type] | None = None, **kwargs: dict[str, Any]) -> None:
        '''
        Generates MPS kernel call code. It should look something like:
        ```
        auto mps_lib_0_func = mps_lib_0.getKernelFunction("generated_kernel");
        auto mps_lib_0_func_handle = AOTIMetalKernelFunctionHandle(mps_lib_0_func.get());
        mps_lib_0_func->runCommandBlock([&] {
            mps_lib_0_func->startEncoding();
            aoti_torch_mps_set_arg(mps_lib_0_func_handle, 0, buf0);
            aoti_torch_mps_set_arg(mps_lib_0_func_handle, 1, arg0_1);
            ...
            mps_lib_0_func->dispatch(9);
        });
        ```
        '''
    def wrap_kernel_call(self, name: str, call_args: list[str]) -> str: ...
    @staticmethod
    def get_device_include_path(device: str) -> str: ...
