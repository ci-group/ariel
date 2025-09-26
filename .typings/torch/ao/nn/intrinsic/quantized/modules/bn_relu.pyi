import torch.ao.nn.intrinsic
import torch.ao.nn.quantized as nnq

__all__ = ['BNReLU2d', 'BNReLU3d']

class BNReLU2d(nnq.BatchNorm2d):
    """
    A BNReLU2d module is a fused module of BatchNorm2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm2d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm2d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU2d
    def __init__(self, num_features, eps: float = 1e-05, momentum: float = 0.1, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point): ...

class BNReLU3d(nnq.BatchNorm3d):
    """
    A BNReLU3d module is a fused module of BatchNorm3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm3d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm3d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU3d
    def __init__(self, num_features, eps: float = 1e-05, momentum: float = 0.1, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point): ...
