import torch.nn as nn
from _typeshed import Incomplete

__all__ = ['Linear']

class Linear(nn.Linear):
    """
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Linear
    qconfig: Incomplete
    weight_fake_quant: Incomplete
    def __init__(self, in_features, out_features, bias: bool = True, qconfig=None, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a qat module from a float module or qparams_dict
        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
    def to_float(self): ...
