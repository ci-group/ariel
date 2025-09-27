from .. import config as config
from ..codecache import CodeCacheFuture as CodeCacheFuture, code_hash as code_hash, get_path as get_path, write_atomic as write_atomic
from ..runtime.benchmarking import benchmarker as benchmarker
from ..utils import IndentedBuffer as IndentedBuffer, cache_on_self as cache_on_self
from ..virtualized import V as V
from .common import TensorArg as TensorArg, WorkspaceArg as WorkspaceArg
from _typeshed import Incomplete
from torch._inductor.metrics import get_metric_table as get_metric_table, is_metric_table_enabled as is_metric_table_enabled
from torch.utils._ordered_set import OrderedSet as OrderedSet

log: Incomplete

class MultiKernelState:
    """
    Maintain state of multi-kernel compilation so we don't define duplicated
    multi-kernel for the same set of sub-kernels.

    V.graph.wrapper_code has a reference to MultiKernelState instance.
    """
    subkernel_to_kernel_name: Incomplete
    kernel_defs: Incomplete
    def __init__(self) -> None: ...
    def define_kernel(self, kernels):
        '''
        Previously we name the multi kernel as "multi_kernel_{kernel_names[0]}".
        This has some minor issue.

        E.g. for persistent reduction https://gist.github.com/shunting314/39e7c00ff8bb2055942ed5a3255d61ca ,
        there are 2 flavors of non-persistent reduction:
          https://gist.github.com/shunting314/056d43d35907e87efb883970b35c17d4
        and
          https://gist.github.com/shunting314/02ee753b65c513c54e695626afe682bd

        The only different is cache eviction policy.

        We should name the multi-kernel differently in these 2 cases.
        '''

class MultiKernel:
    """
    This class maintains the compile time state for multi kernels.

    Assume we do codegen for a MultiKernel encapsulating kernel1 and kernel2.
    The generated definition for the multi-kernel will looks like:
    ```
    multi_kernel_kernel1 = MultiKernelCall(
        [kernel1, kernel2], multi_kernel_definition_code
    )
    ```

    Here is a concrete example: https://gist.github.com/shunting314/d9f3fb6bc6cee3dbae005825ca196d39
    """
    kernels: Incomplete
    kernel_name: Incomplete
    args: Incomplete
    def __init__(self, kernels) -> None: ...
    @staticmethod
    def _merge_workspace_args(left: list[WorkspaceArg], right: list[WorkspaceArg]): ...
    @staticmethod
    def merge_workspaces_inplace(kernels): ...
    def call_kernel(self, kernel_name) -> None:
        """
        Collect the union of arguments from all subkernels as the arguments
        for the multi-kernel.
        """
    def codegen_nan_check(self) -> None: ...
    @property
    def removed_buffers(self): ...
    @property
    def inplaced_to_remove(self): ...
    @property
    @cache_on_self
    def inplace_update_buffers(self):
        """
        Make sure all kernels have the same inplace update mappings.
        """
    def warn_mix_layout(self, kernel_name: str): ...

class MultiKernelCall:
    """
    This class is called at run time to actually run the kernel
    """
    _kernels: Incomplete
    multi_kernel_name: Incomplete
    disable_cache: Incomplete
    picked_kernel: Incomplete
    _recorded: bool
    def __init__(self, multi_kernel_name, kernels) -> None: ...
    def cache_file_path(self): ...
    def load_cache(self) -> None: ...
    def store_cache(self) -> None: ...
    @property
    def kernels(self):
        """
        Read results from future.

        This should be called after parallel compilation is done.
        In case you call this before compilation is done,
        it may slow down the parallel compilation.
        """
    def benchmark_sub_kernels(self, *args, **kwargs):
        """
        Benchmark all the sub kernels and return the execution time
        (in milliseconds) for each of time.

        Unit test may mock this method to force a specific kernel to
        be picked.
        """
    @staticmethod
    def record_choice(multi_kernel_name: str, picked_kernel_name: str):
        """
        Record the multi-kernel choice for cpp-wrapper after autotuning

        We should do nothing if this function is not called during codegen.
        """
    @staticmethod
    def lookup_choice(multi_kernel_name: str) -> str: ...
    def run(self, *args, **kwargs) -> None: ...
    def _metrics_table_row(self, timings): ...
