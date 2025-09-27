from torch import nn
from typing import Any

__all__ = ['module_contains_param', 'swap_module', 'module_to_fqn', 'fqn_to_module', 'get_arg_info_from_tensor_fqn', 'FakeSparsity']

def module_contains_param(module: nn.Module, parametrization: type[nn.Module]) -> bool: ...
def swap_module(mod: nn.Module, mapping: dict[type[nn.Module], type[nn.Module]]) -> nn.Module:
    """Swaps the module using from_dense according to the mapping passed in.
    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to sparse nn module
    Return:
        The corresponding sparse module of `mod` according to mapping, created using from_dense
    """
def module_to_fqn(model: nn.Module, module: nn.Module, prefix: str = '') -> str | None:
    """
    Returns the fqn for a module or None if module not a descendent of model.
    """
def fqn_to_module(model: nn.Module | None, path: str) -> nn.Module | None:
    """
    Given an fqn, returns the corresponding module or tensor or None if the fqn given by `path`
    doesn't correspond to anything. Similar to model.get_submodule(path) but works for tensors.
    """
def get_arg_info_from_tensor_fqn(model: nn.Module, tensor_fqn: str) -> dict[str, Any]:
    """
    Uses tensor_fqn to obtain a dict containing module_fqn, module and tensor_name
    """

class FakeSparsity(nn.Module):
    """Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """
    def __init__(self, mask) -> None: ...
    def forward(self, x): ...
    def state_dict(self, *args, **kwargs): ...
