import torch.nn as nn
from torch.ao.quantization.utils import Pattern
from typing import Callable

__all__ = ['fuse_conv_bn', 'fuse_conv_bn_relu', 'fuse_linear_bn', 'fuse_convtranspose_bn', 'get_fuser_method', 'get_fuser_method_new']

def fuse_conv_bn(is_qat, conv, bn):
    """Return the fused the conv and bn modules.
    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn(m1, b1)
    """
def fuse_conv_bn_relu(is_qat, conv, bn, relu):
    """Return the fused conv and bv modules.

    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> r1 = nn.ReLU(inplace=False)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn_relu(m1, b1, r1)
    """
def fuse_linear_bn(is_qat, linear, bn):
    """Return the fused linear and bn modules.
    Given the linear and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer

    Examples::

        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_linear_bn(m1, b1)
    """
def fuse_convtranspose_bn(is_qat, convt, bn):
    """Return the fused ConvTranspose and bn modules.
    Given ConvTranspose and bn modules, fuses them and returns the fused module

    Args:
        convt: Module instance of type ConvTransposeNd
        bn: BatchNormNd instance that needs to be fused with the linear layer.
            batch norm N should match the ConvTranspose N

    Examples::

        >>> m1 = nn.ConvTranspose2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_convtranspose_bn(m1, b1)
    """
def get_fuser_method(op_list, additional_fuser_method_mapping=None):
    """Get fuser method for the given list of module types.

    Get fuser method for the given list of module types,
    return None if fuser method does not exist
    """
def get_fuser_method_new(op_pattern: Pattern, fuser_method_mapping: dict[Pattern, nn.Sequential | Callable]):
    """Get fuser method.

    This will be made default after we deprecate the get_fuser_method
    Would like to implement this first and have a separate PR for deprecation
    """
