from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.distributed.optim._deprecation_warning import _scripted_functional_optimizer_deprecation_warning as _scripted_functional_optimizer_deprecation_warning

__all__: list[str]

class _FunctionalRprop:
    defaults: Incomplete
    etas: Incomplete
    step_sizes: Incomplete
    foreach: Incomplete
    maximize: Incomplete
    param_group: Incomplete
    state: Incomplete
    def __init__(self, params: list[Tensor], lr: float = 0.01, etas: tuple[float, float] = (0.5, 1.2), step_sizes: tuple[float, float] = (1e-06, 50), foreach: bool = False, maximize: bool = False, _allow_empty_param_list: bool = False) -> None: ...
    def step(self, gradients: list[Tensor | None]): ...
