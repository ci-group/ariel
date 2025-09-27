import types
from _typeshed import Incomplete
from torch.distributed._tools.mod_tracker import ModTracker
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any

__all__ = ['CommDebugMode']

class _CommModeModuleTracker(ModTracker):
    """
    Inherits ModuleTracker and expands on its functionality to track the
    parameters and sharding information of a model at a module-level
    """
    module_helper_dict: Incomplete
    module_parameters_dict: Incomplete
    module_parents_dict: Incomplete
    register_forward_hook_handles: Incomplete
    parent_dict: Incomplete
    parent_list: Incomplete
    sharding_dict: Incomplete
    activation_checkpointing: bool
    name: str
    def __init__(self) -> None: ...
    def _fw_set_module_hook(self, mod, input, output) -> None:
        """
        Updates the current module after module finishes running and
        all other hooks are resolved
        """
    def _fw_pre_hook(self, mod, input) -> None:
        """
        This function is called before the forward pass of a module. It
        collects the parameters and sharding information of a module and
        stores it in a dictionary.
        """
    def _fw_post_hook(self, mod, input, output) -> None:
        """
        This function is called when the forward pass of a module is called.
        It updates the module tracker and removes the module from parent data
        """
    def _bw_hook(self, mod, output) -> None:
        """
        This function is called when the backward pass of a module is called. It
        updates the current module for backward passes
        """
    _fw_pre_handle: Incomplete
    _fw_post_handle: Incomplete
    _bw_handle: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...
    def print_paramater_info(self) -> None: ...
    def print_sharding_info(self) -> None: ...

class CommDebugMode(TorchDispatchMode):
    """
    :class:`CommDebugMode` is a context manager that counts the number of
    functional collectives within its context. It does this using a
    ``TorchDispatchMode``.

    .. note:: Not all collectives are supported yet.

    Example usage

    .. code-block:: python

        mod = ...
        comm_mode = CommDebugMode()
        with comm_mode:
            mod.sum().backward()
        print(comm_mode.get_comm_counts())
    """
    comm_counts: dict[Any, int]
    comm_module_counts: Incomplete
    comm_module_operation_counts: Incomplete
    comm_registry: Incomplete
    advanced_module_tracker: Incomplete
    def __init__(self) -> None: ...
    def generate_json_dump(self, file_name: str = 'comm_mode_log.json', noise_level: int = 3):
        """
        Creates json file used to build browser visual
        0. prints module-level collective counts
        1. prints dTensor operations not included in trivial operations
        2. prints operations not included in trivial operations
        3. prints all operations
        """
    def generate_comm_debug_tracing_table(self, noise_level: int = 3):
        """
        Generates detailed table displaying operations and collective tracing information
        on a module level. Amount of information is dependent on noise_level

        0. prints module-level collective counts
        1. prints dTensor operations not included in trivial operations, module information
        2. prints operations not included in trivial operations
        3. prints all operations
        """
    def _get_operations_list(self, module_operation_counts): ...
    def get_total_counts(self) -> int: ...
    def get_comm_counts(self) -> dict[Any, int]:
        """Returns the communication counts as a dictionary.

        Returns:
            Dict[Any, int]: The communication counts as a dictionary.
        """
    def get_parameter_info(self) -> dict[str, dict[str, Any]]: ...
    def get_sharding_info(self) -> dict[str, dict[str, Any]]: ...
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...
    def log_comm_debug_tracing_table_to_file(self, file_name: str = 'comm_mode_log.txt', noise_level: int = 3) -> None:
        """
        Alternative to console CommDebugMode output, writes to file specified by the user
        """
    def _set_noise_parameters(self, noise_level):
        """
        sets variables controlling what information displays based on noise level
        """
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...
