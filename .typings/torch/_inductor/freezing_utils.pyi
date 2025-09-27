import contextlib
import torch
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any

_TLS: Incomplete

def _freezing_active() -> bool: ...
@contextlib.contextmanager
def enter_freezing() -> Generator[Any, None, None]:
    """
    Context manager to designate when freezing is active.
    """
def record_has_frozen_params(gm: torch.fx.GraphModule) -> None:
    """
    Mark the gm as having frozen params.
    """
def has_frozen_params(gm: torch.fx.GraphModule) -> bool:
    """
    Return True if the gm has frozen parameters.
    """
def maybe_set_is_frozen_param(t: torch.Tensor) -> None:
    """
    Mark the provided tensor as a frozen param if freezing is active.
    """
def is_frozen_param(t: torch.Tensor) -> bool:
    """
    Return True if the tensor is a frozen param.
    """
