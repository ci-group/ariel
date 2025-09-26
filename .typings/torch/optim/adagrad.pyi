from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor

__all__ = ['Adagrad', 'adagrad']

class Adagrad(Optimizer):
    _need_device_dtype_check_for_fused: bool
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.01, lr_decay: float = 0, weight_decay: float = 0, initial_accumulator_value: float = 0, eps: float = 1e-10, foreach: bool | None = None, *, maximize: bool = False, differentiable: bool = False, fused: bool | None = None) -> None: ...
    def __setstate__(self, state) -> None: ...
    def share_memory(self) -> None: ...
    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def adagrad(params: list[Tensor], grads: list[Tensor], state_sums: list[Tensor], state_steps: list[Tensor], fused: bool | None = None, grad_scale: Tensor | None = None, found_inf: Tensor | None = None, has_sparse_grad: bool = False, foreach: bool | None = None, differentiable: bool = False, has_complex: bool = False, *, lr: float, weight_decay: float, lr_decay: float, eps: float, maximize: bool):
    """Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """
