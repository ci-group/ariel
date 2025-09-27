from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.distributed.optim._deprecation_warning import _scripted_functional_optimizer_deprecation_warning as _scripted_functional_optimizer_deprecation_warning

__all__: list[str]

class _FunctionalRMSprop:
    defaults: Incomplete
    centered: Incomplete
    foreach: Incomplete
    maximize: Incomplete
    param_group: Incomplete
    state: Incomplete
    def __init__(self, params: list[Tensor], lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-08, weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False, foreach: bool = False, maximize: bool = False, _allow_empty_param_list: bool = False) -> None: ...
    def step(self, gradients: list[Tensor | None]): ...
