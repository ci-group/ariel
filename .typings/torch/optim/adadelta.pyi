from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor
from typing import Any

__all__ = ['Adadelta', 'adadelta']

class Adadelta(Optimizer):
    def __init__(self, params: ParamsT, lr: float | Tensor = 1.0, rho: float = 0.9, eps: float = 1e-06, weight_decay: float = 0, foreach: bool | None = None, *, capturable: bool = False, maximize: bool = False, differentiable: bool = False) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group: dict[str, Any], params_with_grad: list[Tensor], grads: list[Tensor], square_avgs: list[Tensor], acc_deltas: list[Tensor], state_steps: list[Tensor]): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def adadelta(params: list[Tensor], grads: list[Tensor], square_avgs: list[Tensor], acc_deltas: list[Tensor], state_steps: list[Tensor], capturable: bool = False, foreach: bool | None = None, differentiable: bool = False, has_complex: bool = False, *, lr: float, rho: float, eps: float, weight_decay: float, maximize: bool):
    """Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """
