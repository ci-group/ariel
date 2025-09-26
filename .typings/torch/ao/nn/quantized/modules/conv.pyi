import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn
from .utils import WeightedQuantizedModule
from _typeshed import Incomplete
from torch.nn.common_types import _size_1_t
from typing import ClassVar

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']

class _ConvNd(WeightedQuantizedModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    in_channels: Incomplete
    out_channels: Incomplete
    kernel_size: Incomplete
    stride: Incomplete
    padding: Incomplete
    dilation: Incomplete
    transposed: Incomplete
    output_padding: Incomplete
    groups: Incomplete
    padding_mode: Incomplete
    scale: float
    zero_point: int
    def _init(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def set_weight_bias(self, qweight, bias_float) -> None: ...
    def bias(self) -> None: ...
    def _weight_bias(self) -> None: ...
    def extra_repr(self): ...
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None: ...
    @torch.jit.export
    def __getstate__(self): ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    training: Incomplete
    @torch.jit.export
    def __setstate__(self, state) -> None: ...
    def __deepcopy__(self, memo): ...
    def __copy__(self): ...
    @classmethod
    def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
        """Creates a qconv object and returns it."""
    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        """Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconv (Module): a reference quantized  module, either produced by torch.ao.quantization
                                utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """

class Conv1d(_ConvNd):
    """Applies a 1D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
        ...                                     dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]]
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_ADD_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nn.Module] | None]
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: _size_1_t = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    _packed_params: Incomplete
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def _weight_bias(self): ...
    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """

class Conv2d(_ConvNd):
    """Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]]
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_ADD_MODULE: ClassVar[type[nni.ConvAdd2d]]
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nni.ConvAddReLU2d]]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    _packed_params: Incomplete
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def _weight_bias(self): ...
    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """

class Conv3d(_ConvNd):
    """Applies a 3D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]]
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_ADD_MODULE: ClassVar[type[nn.Module] | None]
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nn.Module] | None]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    _packed_params: Incomplete
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def _weight_bias(self): ...
    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """

class _ConvTransposeNd(_ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, device=None, dtype=None) -> None: ...
    def _input_padding(self, kernel_size: list[int], dilation: list[int], padding: list[int]) -> list[int]: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Creates a quantized module from a float module or qparams_dict.
        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
    @staticmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):
        """Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconvt (Module): a reference quantized  module, either produced by torch.ao.quantization
                                 utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """

class ConvTranspose1d(_ConvTransposeNd):
    '''Applies a 1D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    .. note:: Currently only the QNNPACK engine is implemented.
        Please, set the `torch.backends.quantized.engine = \'qnnpack\'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = \'qnnpack\'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    '''
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose1d]]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    _packed_params: Incomplete
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def _weight_bias(self): ...
    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...

class ConvTranspose2d(_ConvTransposeNd):
    '''Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # QNNPACK or FBGEMM as backend
        >>> torch.backends.quantized.engine = \'qnnpack\'
        >>> # With square kernels and equal stride
        >>> import torch.ao.nn.quantized as nnq
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    '''
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose2d]]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    _packed_params: Incomplete
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def _weight_bias(self): ...
    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...

class ConvTranspose3d(_ConvTransposeNd):
    '''Applies a 3D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    .. note:: Currently only the FBGEMM engine is implemented.
        Please, set the `torch.backends.quantized.engine = \'fbgemm\'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = \'fbgemm\'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> input = torch.randn(20, 16, 50, 100, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    '''
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose3d]]
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    _packed_params: Incomplete
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def _weight_bias(self): ...
    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...
