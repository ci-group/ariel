import torch.ao.nn.quantized as nnq
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t
from typing import ClassVar

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']

class Conv1d(nnq.Conv1d):
    """A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d` and :class:`~torch.ao.nn.quantized.dynamic.Conv1d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.quantized.dynamic.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> output = m(input)

    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]]
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None]
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, reduce_range: bool = True) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor: ...

class Conv2d(nnq.Conv2d):
    """A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d` and :class:`~torch.ao.nn.quantized.dynamic.Conv2d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]]
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor: ...

class Conv3d(nnq.Conv3d):
    """A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d` and :class:`~torch.ao.nn.quantized.dynamic.Conv3d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> output = m(input)

    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]]
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor: ...

class ConvTranspose1d(nnq.ConvTranspose1d):
    """A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nndq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nndq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nndq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nndq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose1d]]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor: ...

class ConvTranspose2d(nnq.ConvTranspose2d):
    """A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose2d]]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor: ...

class ConvTranspose3d(nnq.ConvTranspose3d):
    """A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose3d]]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor: ...
