from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor

__all__ = ['Adam', 'adam']

class Adam(Optimizer):
    _step_supports_amp_scaling: bool
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.001, betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, amsgrad: bool = False, *, foreach: bool | None = None, maximize: bool = False, capturable: bool = False, differentiable: bool = False, fused: bool | None = None, decoupled_weight_decay: bool = False) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def adam(params: list[Tensor], grads: list[Tensor], exp_avgs: list[Tensor], exp_avg_sqs: list[Tensor], max_exp_avg_sqs: list[Tensor], state_steps: list[Tensor], foreach: bool | None = None, capturable: bool = False, differentiable: bool = False, fused: bool | None = None, grad_scale: Tensor | None = None, found_inf: Tensor | None = None, has_complex: bool = False, decoupled_weight_decay: bool = False, *, amsgrad: bool, beta1: float, beta2: float, lr: float | Tensor, weight_decay: float, eps: float, maximize: bool):
    """Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
