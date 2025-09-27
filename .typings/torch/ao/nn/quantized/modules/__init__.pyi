import torch
from .activation import ELU as ELU, Hardswish as Hardswish, LeakyReLU as LeakyReLU, MultiheadAttention as MultiheadAttention, PReLU as PReLU, ReLU6 as ReLU6, Sigmoid as Sigmoid, Softmax as Softmax
from .batchnorm import BatchNorm2d as BatchNorm2d, BatchNorm3d as BatchNorm3d
from .conv import Conv1d as Conv1d, Conv2d as Conv2d, Conv3d as Conv3d, ConvTranspose1d as ConvTranspose1d, ConvTranspose2d as ConvTranspose2d, ConvTranspose3d as ConvTranspose3d
from .dropout import Dropout as Dropout
from .embedding_ops import Embedding as Embedding, EmbeddingBag as EmbeddingBag
from .functional_modules import FXFloatFunctional as FXFloatFunctional, FloatFunctional as FloatFunctional, QFunctional as QFunctional
from .linear import Linear as Linear
from .normalization import GroupNorm as GroupNorm, InstanceNorm1d as InstanceNorm1d, InstanceNorm2d as InstanceNorm2d, InstanceNorm3d as InstanceNorm3d, LayerNorm as LayerNorm
from .rnn import LSTM as LSTM
from _typeshed import Incomplete

__all__ = ['BatchNorm2d', 'BatchNorm3d', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'DeQuantize', 'ELU', 'Embedding', 'EmbeddingBag', 'GroupNorm', 'Hardswish', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d', 'LayerNorm', 'LeakyReLU', 'Linear', 'LSTM', 'MultiheadAttention', 'Quantize', 'ReLU6', 'Sigmoid', 'Softmax', 'Dropout', 'PReLU', 'FloatFunctional', 'FXFloatFunctional', 'QFunctional']

class Quantize(torch.nn.Module):
    """Quantizes an incoming tensor

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor
     `factory_kwargs`: Dictionary of kwargs used for configuring initialization
         of internal buffers. Currently, `device` and `dtype` are supported.
         Example: `factory_kwargs={'device': 'cuda', 'dtype': torch.float64}`
         will initialize internal buffers as type `torch.float64` on the current CUDA device.
         Note that `dtype` only applies to floating-point buffers.

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> qt = qm(t)
        >>> print(qt)
        tensor([[ 1., -1.],
                [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """
    scale: torch.Tensor
    zero_point: torch.Tensor
    dtype: Incomplete
    def __init__(self, scale, zero_point, dtype, factory_kwargs=None) -> None: ...
    def forward(self, X): ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant: bool = False): ...
    def extra_repr(self): ...

class DeQuantize(torch.nn.Module):
    """Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """
    def forward(self, Xq): ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant: bool = False): ...
