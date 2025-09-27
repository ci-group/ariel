from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor

__all__ = ['RMSprop', 'rmsprop']

class RMSprop(Optimizer):
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.01, alpha: float = 0.99, eps: float = 1e-08, weight_decay: float = 0, momentum: float = 0, centered: bool = False, capturable: bool = False, foreach: bool | None = None, maximize: bool = False, differentiable: bool = False) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group, params_with_grad, grads, square_avgs, momentum_buffer_list, grad_avgs, state_steps): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def rmsprop(params: list[Tensor], grads: list[Tensor], square_avgs: list[Tensor], grad_avgs: list[Tensor], momentum_buffer_list: list[Tensor], state_steps: list[Tensor], foreach: bool | None = None, maximize: bool = False, differentiable: bool = False, capturable: bool = False, has_complex: bool = False, *, lr: float, alpha: float, eps: float, weight_decay: float, momentum: float, centered: bool):
    """Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """
