from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor

__all__ = ['ASGD', 'asgd']

class ASGD(Optimizer):
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.01, lambd: float = 0.0001, alpha: float = 0.75, t0: float = 1000000.0, weight_decay: float = 0, foreach: bool | None = None, maximize: bool = False, differentiable: bool = False, capturable: bool = False) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group, params_with_grad, grads, mus, axs, etas, state_steps): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def asgd(params: list[Tensor], grads: list[Tensor], axs: list[Tensor], mus: list[Tensor], etas: list[Tensor], state_steps: list[Tensor], foreach: bool | None = None, maximize: bool = False, differentiable: bool = False, capturable: bool = False, has_complex: bool = False, *, lambd: float, lr: float, t0: float, alpha: float, weight_decay: float):
    """Functional API that performs asgd algorithm computation.

    See :class:`~torch.optim.ASGD` for details.
    """
