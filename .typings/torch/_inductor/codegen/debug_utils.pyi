import functools
import types
from .. import config as config
from ..virtualized import V as V
from .multi_kernel import MultiKernel as MultiKernel
from _typeshed import Incomplete
from enum import Enum
from typing import Callable

log: Incomplete

def _print_debugging_tensor_value_info(msg, arg) -> None: ...

class IntermediateValueDebuggingLevel(Enum):
    OFF = '0'
    SAVE_ONLY = '1'
    PRINT_ONLY = '2'
    PRINT_KERNEL_NAMES_ONLY = '3'

class DebugPrinterManager:
    debug_printer_level: Incomplete
    use_array_ref: Incomplete
    args_to_print_or_save: Incomplete
    kernel_name: Incomplete
    arg_signatures: list[type] | None
    kernel: Incomplete
    filtered_kernel_names_to_print: Incomplete
    kernel_type: Incomplete
    def __init__(self, debug_printer_level, use_array_ref: bool, writeline: Callable[..., None] | None = None, args_to_print_or_save: list[str] | None = None, kernel_name: str = '', kernel=None, arg_signatures: list[type] | None = None, kernel_type=None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, args_to_print_or_save: type[BaseException] | None, kernel_name: BaseException | None, arg_signatures: types.TracebackType | None) -> None: ...
    def _perform_debug_print_or_save_helper(self, args_to_print_or_save, kernel_name, before_launch, arg_signatures: list[type] | None = None): ...
    @functools.lru_cache
    def _get_debug_filtered_kernel_names(self) -> list[str]: ...
    def set_printer_args(self, args_to_print_or_save: list[str], kernel_name: str, arg_signatures: list[type] | None, kernel, kernel_type=None): ...
    def codegen_model_inputs_value_print(self, input_args_to_print: list[str]) -> None: ...
    def codegen_intermediate_tensor_value_save(self, args_to_save, kernel_name, before_launch: bool = True, arg_signatures: list[type] | None = None) -> None: ...
    def codegen_intermediate_tensor_value_print(self, args_to_print, kernel_name, before_launch: bool = True, arg_signatures: list[type] | None = None) -> None: ...
