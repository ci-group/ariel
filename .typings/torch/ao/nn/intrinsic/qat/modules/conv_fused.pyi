import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn as nn
from _typeshed import Incomplete
from typing import ClassVar

__all__ = ['ConvBn1d', 'ConvBnReLU1d', 'ConvReLU1d', 'ConvBn2d', 'ConvBnReLU2d', 'ConvReLU2d', 'ConvBn3d', 'ConvBnReLU3d', 'ConvReLU3d', 'update_bn_stats', 'freeze_bn_stats']

class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):
    _version: int
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]
    qconfig: Incomplete
    freeze_bn: Incomplete
    bn: Incomplete
    weight_fake_quant: Incomplete
    bias: Incomplete
    _enable_slow_path_for_better_numerical_stability: bool
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, eps: float = 1e-05, momentum: float = 0.1, freeze_bn: bool = False, qconfig=None, dim: int = 2) -> None: ...
    def reset_running_stats(self) -> None: ...
    def reset_bn_parameters(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def update_bn_stats(self): ...
    def freeze_bn_stats(self): ...
    def _forward(self, input): ...
    def _forward_approximate(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
    def _forward_slow(self, input):
        """
        A more accurate but slow method to compute conv bn fusion, following https://arxiv.org/pdf/1806.08342.pdf
        It requires two forward passes but handles the case bn.weight == 0

        Conv: Y = WX + B_c
        Conv without bias: Y0 = WX = Y - B_c, Y = Y0 + B_c

        Batch statistics:
          mean_Y = Y.mean()
                 = Y0.mean() + B_c
          var_Y = (Y - mean_Y)^2.mean()
                = (Y0 - Y0.mean())^2.mean()
        BN (r: bn.weight, beta: bn.bias):
          Z = r * (Y - mean_Y) / sqrt(var_Y + eps) + beta
            = r * (Y0 - Y0.mean()) / sqrt(var_Y + eps) + beta

        Fused Conv BN training (std_Y = sqrt(var_Y + eps)):
          Z = (r * W / std_Y) * X + r * (B_c - mean_Y) / std_Y + beta
            = (r * W / std_Y) * X - r * Y0.mean() / std_Y + beta

        Fused Conv BN inference (running_std = sqrt(running_var + eps)):
          Z = (r * W / running_std) * X - r * (running_mean - B_c) / running_std + beta

        QAT with fused conv bn:
          Z_train = fake_quant(r * W / running_std) * X * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
                  = conv(X, fake_quant(r * W / running_std)) * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
          Z_inference = conv(X, fake_quant(r * W / running_std)) - r * (running_mean - B_c) / running_std + beta
        """
    def extra_repr(self): ...
    def forward(self, input): ...
    training: Incomplete
    def train(self, mode: bool = True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
    def to_float(self): ...

class ConvBn1d(_ConvBnNd, nn.Conv1d):
    """
    A ConvBn1d module is a module fused from Conv1d and BatchNorm1d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Conv1d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm1d]]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None]
    _FLOAT_MODULE: ClassVar[type[nn.Module]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias=None, padding_mode: str = 'zeros', eps: float = 1e-05, momentum: float = 0.1, freeze_bn: bool = False, qconfig=None) -> None: ...

class ConvBnReLU1d(ConvBn1d):
    """
    A ConvBnReLU1d module is a module fused from Conv1d, BatchNorm1d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE: ClassVar[type[nn.Module]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]]
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm1d]]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None]
    _FUSED_FLOAT_MODULE: ClassVar[type[nn.Module] | None]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias=None, padding_mode: str = 'zeros', eps: float = 1e-05, momentum: float = 0.1, freeze_bn: bool = False, qconfig=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class ConvReLU1d(nnqat.Conv1d, nni._FusedModule):
    """A ConvReLU1d module is a fused module of Conv1d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv1d` and
    :class:`~torch.nn.BatchNorm1d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE: ClassVar[type[nni.ConvReLU1d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]]
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None]
    qconfig: Incomplete
    weight_fake_quant: Incomplete
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', qconfig=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class ConvBn2d(_ConvBnNd, nn.Conv2d):
    """
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE: ClassVar[type[nni.ConvBn2d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]]
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias=None, padding_mode: str = 'zeros', eps: float = 1e-05, momentum: float = 0.1, freeze_bn: bool = False, qconfig=None) -> None: ...

class ConvBnReLU2d(ConvBn2d):
    """
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE: ClassVar[type[nni.ConvBnReLU2d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]]
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm2d]]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None]
    _FUSED_FLOAT_MODULE: ClassVar[type[nni.ConvReLU2d] | None]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias=None, padding_mode: str = 'zeros', eps: float = 1e-05, momentum: float = 0.1, freeze_bn: bool = False, qconfig=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class ConvReLU2d(nnqat.Conv2d, nni._FusedModule):
    """A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE: ClassVar[type[nn.Module]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]]
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None]
    qconfig: Incomplete
    weight_fake_quant: Incomplete
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', qconfig=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class ConvBn3d(_ConvBnNd, nn.Conv3d):
    """
    A ConvBn3d module is a module fused from Conv3d and BatchNorm3d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d`.

    Similar to :class:`torch.nn.Conv3d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE: ClassVar[type[nni.ConvBn3d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]]
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias=None, padding_mode: str = 'zeros', eps: float = 1e-05, momentum: float = 0.1, freeze_bn: bool = False, qconfig=None) -> None: ...

class ConvBnReLU3d(ConvBn3d):
    """
    A ConvBnReLU3d module is a module fused from Conv3d, BatchNorm3d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE: ClassVar[type[nni.ConvBnReLU3d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]]
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm3d]]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.ReLU] | None]
    _FUSED_FLOAT_MODULE: ClassVar[type[nni.ConvReLU3d] | None]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias=None, padding_mode: str = 'zeros', eps: float = 1e-05, momentum: float = 0.1, freeze_bn: bool = False, qconfig=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class ConvReLU3d(nnqat.Conv3d, nni._FusedModule):
    """A ConvReLU3d module is a fused module of Conv3d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv3d` and
    :class:`~torch.nn.BatchNorm3d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE: ClassVar[type[nni.ConvReLU3d]]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]]
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None]
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None]
    qconfig: Incomplete
    weight_fake_quant: Incomplete
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', qconfig=None) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

def update_bn_stats(mod) -> None: ...
def freeze_bn_stats(mod) -> None: ...
