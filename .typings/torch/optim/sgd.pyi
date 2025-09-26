from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor

__all__ = ['SGD', 'sgd']

class SGD(Optimizer):
    _step_supports_amp_scaling: bool
    _need_device_dtype_check_for_fused: bool
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.001, momentum: float = 0, dampening: float = 0, weight_decay: float | Tensor = 0, nesterov: bool = False, *, maximize: bool = False, foreach: bool | None = None, differentiable: bool = False, fused: bool | None = None) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group, params, grads, momentum_buffer_list): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def sgd(params: list[Tensor], d_p_list: list[Tensor], momentum_buffer_list: list[Tensor | None], has_sparse_grad: bool = False, foreach: bool | None = None, fused: bool | None = None, grad_scale: Tensor | None = None, found_inf: Tensor | None = None, *, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool, maximize: bool):
    """Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
