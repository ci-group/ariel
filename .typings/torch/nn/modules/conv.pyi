from .lazy import LazyModuleMixin
from .module import Module
from _typeshed import Incomplete
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parameter import UninitializedParameter

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'LazyConv1d', 'LazyConv2d', 'LazyConv3d', 'LazyConvTranspose1d', 'LazyConvTranspose2d', 'LazyConvTranspose3d']

class _ConvNd(Module):
    __constants__: Incomplete
    __annotations__: Incomplete
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor: ...
    in_channels: int
    _reversed_padding_repeated_twice: list[int]
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Tensor | None
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, ...], stride: tuple[int, ...], padding: str | tuple[int, ...], dilation: tuple[int, ...], transposed: bool, output_padding: tuple[int, ...], groups: int, bias: bool, padding_mode: str, device=None, dtype=None) -> None: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self): ...
    def __setstate__(self, state) -> None: ...

class Conv1d(_ConvNd):
    __doc__: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: str | _size_1_t = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor | None): ...
    def forward(self, input: Tensor) -> Tensor: ...

class Conv2d(_ConvNd):
    __doc__: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: str | _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor | None): ...
    def forward(self, input: Tensor) -> Tensor: ...

class Conv3d(_ConvNd):
    __doc__: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: str | _size_3_t = 0, dilation: _size_3_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor | None): ...
    def forward(self, input: Tensor) -> Tensor: ...

class _ConvTransposeNd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, device=None, dtype=None) -> None: ...
    def _output_padding(self, input: Tensor, output_size: list[int] | None, stride: list[int], padding: list[int], kernel_size: list[int], num_spatial_dims: int, dilation: list[int] | None = None) -> list[int]: ...

class ConvTranspose1d(_ConvTransposeNd):
    __doc__: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, output_padding: _size_1_t = 0, groups: int = 1, bias: bool = True, dilation: _size_1_t = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor: ...

class ConvTranspose2d(_ConvTransposeNd):
    __doc__: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t = 0, output_padding: _size_2_t = 0, groups: int = 1, bias: bool = True, dilation: _size_2_t = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor:
        """
        Performs the forward pass.

        Attributes:
            input (Tensor): The input tensor.
            output_size (list[int], optional): A list of integers representing
                the size of the output tensor. Default is None.
        """

class ConvTranspose3d(_ConvTransposeNd):
    __doc__: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _size_3_t = 0, output_padding: _size_3_t = 0, groups: int = 1, bias: bool = True, dilation: _size_3_t = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor: ...

class _ConvTransposeMixin(_ConvTransposeNd):
    def __init__(self, *args, **kwargs) -> None: ...

class _LazyConvXdMixin(LazyModuleMixin):
    groups: int
    transposed: bool
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    weight: UninitializedParameter
    bias: UninitializedParameter
    def reset_parameters(self) -> None: ...
    def initialize_parameters(self, input: Tensor, *args, **kwargs) -> None: ...
    def _get_in_channels(self, input: Tensor) -> int: ...
    def _get_num_spatial_dims(self) -> int: ...

class LazyConv1d(_LazyConvXdMixin, Conv1d):
    """A :class:`torch.nn.Conv1d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`Conv1d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

    .. seealso:: :class:`torch.nn.Conv1d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """
    cls_to_become = Conv1d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(self, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_num_spatial_dims(self) -> int: ...

class LazyConv2d(_LazyConvXdMixin, Conv2d):
    """A :class:`torch.nn.Conv2d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`Conv2d` that is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

    .. seealso:: :class:`torch.nn.Conv2d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """
    cls_to_become = Conv2d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(self, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_num_spatial_dims(self) -> int: ...

class LazyConv3d(_LazyConvXdMixin, Conv3d):
    """A :class:`torch.nn.Conv3d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`Conv3d` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

    .. seealso:: :class:`torch.nn.Conv3d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """
    cls_to_become = Conv3d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(self, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _size_3_t = 0, dilation: _size_3_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_num_spatial_dims(self) -> int: ...

class LazyConvTranspose1d(_LazyConvXdMixin, ConvTranspose1d):
    """A :class:`torch.nn.ConvTranspose1d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose1d` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose1d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """
    cls_to_become = ConvTranspose1d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(self, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, output_padding: _size_1_t = 0, groups: int = 1, bias: bool = True, dilation: _size_1_t = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_num_spatial_dims(self) -> int: ...

class LazyConvTranspose2d(_LazyConvXdMixin, ConvTranspose2d):
    """A :class:`torch.nn.ConvTranspose2d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose2d` is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose2d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """
    cls_to_become = ConvTranspose2d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(self, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t = 0, output_padding: _size_2_t = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_num_spatial_dims(self) -> int: ...

class LazyConvTranspose3d(_LazyConvXdMixin, ConvTranspose3d):
    """A :class:`torch.nn.ConvTranspose3d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose3d` is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose3d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """
    cls_to_become = ConvTranspose3d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(self, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t = 1, padding: _size_3_t = 0, output_padding: _size_3_t = 0, groups: int = 1, bias: bool = True, dilation: _size_3_t = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_num_spatial_dims(self) -> int: ...
