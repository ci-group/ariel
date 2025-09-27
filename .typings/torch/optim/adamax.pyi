from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor

__all__ = ['Adamax', 'adamax']

class Adamax(Optimizer):
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.002, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, foreach: bool | None = None, *, maximize: bool = False, differentiable: bool = False, capturable: bool = False) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_infs, state_steps): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def adamax(params: list[Tensor], grads: list[Tensor], exp_avgs: list[Tensor], exp_infs: list[Tensor], state_steps: list[Tensor], foreach: bool | None = None, maximize: bool = False, differentiable: bool = False, capturable: bool = False, has_complex: bool = False, *, eps: float, beta1: float, beta2: float, lr: float, weight_decay: float):
    """Functional API that performs adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """
