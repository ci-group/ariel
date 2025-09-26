from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable
from torch import Tensor

__all__ = ['Rprop', 'rprop']

class Rprop(Optimizer):
    def __init__(self, params: ParamsT, lr: float | Tensor = 0.01, etas: tuple[float, float] = (0.5, 1.2), step_sizes: tuple[float, float] = (1e-06, 50), *, capturable: bool = False, foreach: bool | None = None, maximize: bool = False, differentiable: bool = False) -> None: ...
    def __setstate__(self, state) -> None: ...
    def _init_group(self, group, params, grads, prevs, step_sizes, state_steps): ...
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def rprop(params: list[Tensor], grads: list[Tensor], prevs: list[Tensor], step_sizes: list[Tensor], state_steps: list[Tensor], foreach: bool | None = None, capturable: bool = False, maximize: bool = False, differentiable: bool = False, has_complex: bool = False, *, step_size_min: float, step_size_max: float, etaminus: float, etaplus: float):
    """Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """
