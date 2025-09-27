import torch
from _typeshed import Incomplete

__all__ = ['model_is_exported']

class _WrapperModule(torch.nn.Module):
    """Class to wrap a callable in an :class:`torch.nn.Module`. Use this if you
    are trying to export a callable.
    """
    fn: Incomplete
    def __init__(self, fn) -> None: ...
    def forward(self, *args, **kwargs):
        """Simple forward that just calls the ``fn`` provided to :meth:`WrapperModule.__init__`."""

def model_is_exported(m: torch.nn.Module) -> bool:
    """
    Return True if the `torch.nn.Module` was exported, False otherwise
    (e.g. if the model was FX symbolically traced or not traced at all).
    """
