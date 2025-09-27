import torch
from typing import TypeVar

__all__ = ['fuse_conv_bn_eval', 'fuse_conv_bn_weights', 'fuse_linear_bn_eval', 'fuse_linear_bn_weights']

ConvT = TypeVar('ConvT', bound='torch.nn.modules.conv._ConvNd')
LinearT = TypeVar('LinearT', bound='torch.nn.Linear')

def fuse_conv_bn_eval(conv: ConvT, bn: torch.nn.modules.batchnorm._BatchNorm, transpose: bool = False) -> ConvT:
    """Fuse a convolutional module and a BatchNorm module into a single, new convolutional module.

    Args:
        conv (torch.nn.modules.conv._ConvNd): A convolutional module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.
        transpose (bool, optional): If True, transpose the convolutional weight. Defaults to False.

    Returns:
        torch.nn.modules.conv._ConvNd: The fused convolutional module.

    .. note::
        Both ``conv`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
def fuse_conv_bn_weights(conv_w: torch.Tensor, conv_b: torch.Tensor | None, bn_rm: torch.Tensor, bn_rv: torch.Tensor, bn_eps: float, bn_w: torch.Tensor | None, bn_b: torch.Tensor | None, transpose: bool = False) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.

    Args:
        conv_w (torch.Tensor): Convolutional weight.
        conv_b (Optional[torch.Tensor]): Convolutional bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (Optional[torch.Tensor]): BatchNorm weight.
        bn_b (Optional[torch.Tensor]): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused convolutional weight and bias.
    """
def fuse_linear_bn_eval(linear: LinearT, bn: torch.nn.modules.batchnorm._BatchNorm) -> LinearT:
    """Fuse a linear module and a BatchNorm module into a single, new linear module.

    Args:
        linear (torch.nn.Linear): A Linear module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.

    Returns:
        torch.nn.Linear: The fused linear module.

    .. note::
        Both ``linear`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
def fuse_linear_bn_weights(linear_w: torch.Tensor, linear_b: torch.Tensor | None, bn_rm: torch.Tensor, bn_rv: torch.Tensor, bn_eps: float, bn_w: torch.Tensor, bn_b: torch.Tensor) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.

    Args:
        linear_w (torch.Tensor): Linear weight.
        linear_b (Optional[torch.Tensor]): Linear bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (torch.Tensor): BatchNorm weight.
        bn_b (torch.Tensor): BatchNorm bias.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused linear weight and bias.
    """
