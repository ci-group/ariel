import torch.nn as nn
from .contract import _State as _State, contract as contract
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from torch.utils.checkpoint import _DEFAULT_DETERMINISM_MODE as _DEFAULT_DETERMINISM_MODE, _checkpoint_without_reentrant_generator as _checkpoint_without_reentrant_generator

@contextmanager
def _no_hook(module: nn.Module, user_ctx: AbstractContextManager | None = None):
    """
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """

class _CheckpointState(_State):
    enable_hook: bool
    _ac_generator: Generator[None, None, None] | None

def checkpoint(module: nn.Module, **kwargs) -> nn.Module:
    """
    This is a composable activation checkpointing API. Unlike functional
    activation checkpointing APIs, this one does not require changing model
    source code. Unlike ``nn.Module`` wrapper activation checkpointing APIs,
    this one does not modify model structure or fully-qualified names either.
    Under the hood, it registers activation checkpointing logic as pre- and
    post-forward hooks. Hence, this API can be easily applied to any model or
    sub-modules in the model.

    Args:
        module (nn.Module): the target model or sub-module to apply activation
            checkpointing.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> model = MyModel()
        >>> checkpoint(model.l1)  # apply activation checkpointing only to l1
        >>> model(torch.zeros(2, 10)).sum().backward()

    """
