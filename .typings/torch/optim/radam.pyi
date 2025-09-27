from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor

__all__ = ['RAdam', 'radam']

class RAdam(Optimizer):
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, decoupled_weight_decay: bool = False, *, foreach: bool | None = None, maximize: bool = False, capturable: bool = False, differentiable: bool = False) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def radam(params: list[Tensor], grads: list[Tensor], exp_avgs: list[Tensor], exp_avg_sqs: list[Tensor], state_steps: list[Tensor], decoupled_weight_decay: bool = False, foreach: bool | None = None, differentiable: bool = False, capturable: bool = False, has_complex: bool = False, maximize: bool = False, *, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float):
    """Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """
