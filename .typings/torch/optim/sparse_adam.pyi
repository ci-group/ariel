from .optimizer import Optimizer, ParamsT
from torch import Tensor

__all__ = ['SparseAdam']

class SparseAdam(Optimizer):
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, maximize: bool = False) -> None: ...
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
