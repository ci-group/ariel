from _typeshed import Incomplete
from torch import nn as nn
from torch.nn.utils.parametrize import is_parametrized as is_parametrized

def module_contains_param(module, parametrization): ...

class FakeStructuredSparsity(nn.Module):
    """
    Parametrization for Structured Pruning. Like FakeSparsity, this should be attached to
    the  'weight' or any other parameter that requires a mask.

    Instead of an element-wise bool mask, this parameterization uses a row-wise bool mask.
    """
    def __init__(self, mask) -> None: ...
    def forward(self, x): ...
    def state_dict(self, *args, **kwargs): ...

class BiasHook:
    param: Incomplete
    prune_bias: Incomplete
    def __init__(self, parametrization, prune_bias) -> None: ...
    def __call__(self, module, input, output): ...
