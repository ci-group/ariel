import torch
from enum import Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType
from typing import AnyStr

class LintCode(Enum):
    BUNDLED_INPUT = 1
    REQUIRES_GRAD = 2
    DROPOUT = 3
    BATCHNORM = 4

def optimize_for_mobile(script_module: torch.jit.ScriptModule, optimization_blocklist: set[MobileOptimizerType] | None = None, preserved_methods: list[AnyStr] | None = None, backend: str = 'CPU') -> torch.jit.RecursiveScriptModule:
    """
    Optimize a torch script module for mobile deployment.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.
        optimization_blocklist: A set with type of MobileOptimizerType. When set is not passed,
            optimization method will run all the optimizer pass; otherwise, optimizer
            method will run the optimization pass that is not included inside optimization_blocklist.
        preserved_methods: A list of methods that needed to be preserved when freeze_module pass is invoked
        backend: Device type to use for running the result model ('CPU'(default), 'Vulkan' or 'Metal').
    Returns:
        A new optimized torch script module
    """
def generate_mobile_module_lints(script_module: torch.jit.ScriptModule):
    """
    Generate a list of lints for a given torch script module.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.

    Returns:
        lint_map: A list of dictionary that contains modules lints
    """
def _get_bundled_inputs_preserved_attributes(script_module: torch.jit.ScriptModule, preserved_methods: list[str]) -> list[str]: ...
