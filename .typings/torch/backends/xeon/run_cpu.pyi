from _typeshed import Incomplete
from torch.distributed.elastic.multiprocessing import Std as Std, start_processes as start_processes

format_str: str
logger: Incomplete

class _CPUinfo:
    """Get CPU information, such as cores list and NUMA information."""
    cpuinfo: Incomplete
    node_nums: Incomplete
    node_physical_cores: list[list[int]]
    node_logical_cores: list[list[int]]
    physical_core_node_map: Incomplete
    logical_core_node_map: Incomplete
    def __init__(self, test_input: str = '') -> None: ...
    def _physical_core_nums(self): ...
    def _logical_core_nums(self): ...
    def get_node_physical_cores(self, node_id): ...
    def get_node_logical_cores(self, node_id): ...
    def get_all_physical_cores(self): ...
    def get_all_logical_cores(self): ...
    def numa_aware_check(self, core_list):
        """
        Check whether all cores in core_list are in the same NUMA node.

        Cross NUMA will reduce performance.
        We strongly advice to not use cores on different nodes.
        """

class _Launcher:
    """Class for launcher."""
    msg_lib_notfound: Incomplete
    cpuinfo: Incomplete
    def __init__(self) -> None: ...
    def add_lib_preload(self, lib_type):
        """Enable TCMalloc/JeMalloc/intel OpenMP."""
    def is_numactl_available(self): ...
    def set_memory_allocator(self, enable_tcmalloc: bool = True, enable_jemalloc: bool = False, use_default_allocator: bool = False) -> None:
        """
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.

        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory reuse and reduce page fault to improve performance.
        """
    def log_env_var(self, env_var_name: str = '') -> None: ...
    def set_env(self, env_name, env_value) -> None: ...
    def set_multi_thread_and_allocator(self, ncores_per_instance, disable_iomp: bool = False, set_kmp_affinity: bool = True, enable_tcmalloc: bool = True, enable_jemalloc: bool = False, use_default_allocator: bool = False) -> None:
        """
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.

        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benefit.
        """
    def launch(self, args) -> None: ...

def _add_memory_allocator_params(parser) -> None: ...
def _add_multi_instance_params(parser) -> None: ...
def _add_kmp_iomp_params(parser) -> None: ...
def create_args(parser=None) -> None:
    """
    Parse the command line options.

    @retval ArgumentParser
    """
def main(args) -> None: ...
