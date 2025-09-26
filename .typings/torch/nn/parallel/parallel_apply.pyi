import torch
from collections.abc import Sequence
from torch.nn.modules import Module
from typing import Any

__all__ = ['get_a_var', 'parallel_apply']

def get_a_var(obj: torch.Tensor | list[Any] | tuple[Any, ...] | dict[Any, Any]) -> torch.Tensor | None: ...
def parallel_apply(modules: Sequence[Module], inputs: Sequence[Any], kwargs_tup: Sequence[dict[str, Any]] | None = None, devices: Sequence[int | torch.device | None] | None = None) -> list[Any]:
    """Apply each `module` in :attr:`modules` in parallel on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
