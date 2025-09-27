from .optimizer import Optimizer, ParamsT
from torch import Tensor

__all__ = ['Adafactor', 'adafactor']

class Adafactor(Optimizer):
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.01, beta2_decay: float = -0.8, eps: tuple[float | None, float] = (None, 0.001), d: float = 1.0, weight_decay: float = 0.0, *, foreach: bool | None = None, maximize: bool = False) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group, params_with_grad, grads, row_vars, col_vars, variances, state_steps): ...
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def adafactor(params: list[Tensor], grads: list[Tensor], row_vars: list[Tensor | None], col_vars: list[Tensor | None], variances: list[Tensor | None], state_steps: list[Tensor], foreach: bool | None = None, grad_scale: Tensor | None = None, found_inf: Tensor | None = None, has_complex: bool = False, *, d: float, lr: float | Tensor, beta2_decay: float, weight_decay: float, eps1: float, eps2: float, maximize: bool):
    """Functional API that performs Adafactor algorithm computation.

    See :class:`~torch.optim.Adafactor` for details.
    """
