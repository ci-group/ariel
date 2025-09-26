from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.distributed.optim._deprecation_warning import _scripted_functional_optimizer_deprecation_warning as _scripted_functional_optimizer_deprecation_warning

__all__: list[str]

class _FunctionalSGD:
    defaults: Incomplete
    nesterov: Incomplete
    maximize: Incomplete
    foreach: Incomplete
    fused: Incomplete
    state: Incomplete
    param_group: Incomplete
    def __init__(self, params: list[Tensor], lr: float = 0.01, momentum: float = 0.0, dampening: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False, maximize: bool = False, foreach: bool = False, fused: bool = False, _allow_empty_param_list: bool = False) -> None: ...
    def step_param(self, param: Tensor, grad: Tensor | None):
        """Similar to self.step, but operates on a single parameter and
        its gradient.
        """
    def step(self, gradients: list[Tensor | None]): ...
