import torch.ao.nn.intrinsic as nni
import torch.nn as nn
from _typeshed import Incomplete

__all__ = ['LinearBn1d']

class LinearBn1d(nn.modules.linear.Linear, nni._FusedModule):
    """
    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached
    with FakeQuantize modules for weight, used in quantization aware training.

    We combined the interface of :class:`torch.nn.Linear` and
    :class:torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    qconfig: Incomplete
    freeze_bn: Incomplete
    bn: Incomplete
    weight_fake_quant: Incomplete
    bias: Incomplete
    def __init__(self, in_features, out_features, bias: bool = True, eps: float = 1e-05, momentum: float = 0.1, freeze_bn: bool = False, qconfig=None) -> None: ...
    def reset_running_stats(self) -> None: ...
    def reset_bn_parameters(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def update_bn_stats(self): ...
    def freeze_bn_stats(self): ...
    def forward(self, input): ...
    training: Incomplete
    def train(self, mode: bool = True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a qat module from a float module or qparams_dict

        Args: `mod' a float module, either produced by torch.ao.quantization
        utilities or directly from user
        """
    def to_float(self): ...
