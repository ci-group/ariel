from .adam import Adam
from .optimizer import ParamsT
from torch import Tensor

__all__ = ['AdamW', 'adamw']

class AdamW(Adam):
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.001, betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, amsgrad: bool = False, *, maximize: bool = False, foreach: bool | None = None, capturable: bool = False, differentiable: bool = False, fused: bool | None = None) -> None: ...
    def __setstate__(self, state) -> None: ...

def adamw(params: list[Tensor], grads: list[Tensor], exp_avgs: list[Tensor], exp_avg_sqs: list[Tensor], max_exp_avg_sqs: list[Tensor], state_steps: list[Tensor], foreach: bool | None = None, capturable: bool = False, differentiable: bool = False, fused: bool | None = None, grad_scale: Tensor | None = None, found_inf: Tensor | None = None, has_complex: bool = False, *, amsgrad: bool, beta1: float, beta2: float, lr: float | Tensor, weight_decay: float, eps: float, maximize: bool):
    """Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
