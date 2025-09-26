import torch
from _typeshed import Incomplete

__all__ = ['LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d']

class LayerNorm(torch.nn.LayerNorm):
    """This is the quantized version of :class:`~torch.nn.LayerNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    weight: Incomplete
    bias: Incomplete
    def __init__(self, normalized_shape, weight, bias, scale, zero_point, eps: float = 1e-05, elementwise_affine: bool = True, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...

class GroupNorm(torch.nn.GroupNorm):
    """This is the quantized version of :class:`~torch.nn.GroupNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    __constants__: Incomplete
    weight: Incomplete
    bias: Incomplete
    def __init__(self, num_groups, num_channels, weight, bias, scale, zero_point, eps: float = 1e-05, affine: bool = True, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class InstanceNorm1d(torch.nn.InstanceNorm1d):
    """This is the quantized version of :class:`~torch.nn.InstanceNorm1d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    weight: Incomplete
    bias: Incomplete
    def __init__(self, num_features, weight, bias, scale, zero_point, eps: float = 1e-05, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...

class InstanceNorm2d(torch.nn.InstanceNorm2d):
    """This is the quantized version of :class:`~torch.nn.InstanceNorm2d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    weight: Incomplete
    bias: Incomplete
    def __init__(self, num_features, weight, bias, scale, zero_point, eps: float = 1e-05, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...

class InstanceNorm3d(torch.nn.InstanceNorm3d):
    """This is the quantized version of :class:`~torch.nn.InstanceNorm3d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    weight: Incomplete
    bias: Incomplete
    def __init__(self, num_features, weight, bias, scale, zero_point, eps: float = 1e-05, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False, device=None, dtype=None) -> None: ...
    def forward(self, input): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...
