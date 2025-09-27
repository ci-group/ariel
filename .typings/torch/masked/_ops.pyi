from torch import Tensor
from torch._prims_common import DimsType
from torch.masked.maskedtensor.core import MaskedTensor
from torch.types import _dtype as DType
from typing import TypeVar
from typing_extensions import ParamSpec, TypeAlias

__all__ = ['sum', 'prod', 'cumsum', 'cumprod', 'amax', 'amin', 'argmax', 'argmin', 'mean', 'median', 'logsumexp', 'norm', 'var', 'std', 'softmax', 'log_softmax', 'softmin', 'normalize']

DimOrDims: TypeAlias = DimsType | None
_T = TypeVar('_T')
_P = ParamSpec('_P')

@_apply_docstring_templates
def sum(input: Tensor | MaskedTensor, dim: DimOrDims = None, *, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
@_apply_docstring_templates
def prod(input: Tensor | MaskedTensor, dim: DimOrDims = None, *, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
@_apply_docstring_templates
def cumsum(input: Tensor, dim: int, *, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
@_apply_docstring_templates
def cumprod(input: Tensor, dim: int, *, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
@_apply_docstring_templates
def amax(input: Tensor | MaskedTensor, dim: DimOrDims = None, *, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}

{reduction_descr}

{reduction_identity_dtype}

{reduction_args}

{reduction_example}"""
@_apply_docstring_templates
def amin(input: Tensor | MaskedTensor, dim: DimOrDims = None, *, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}

{reduction_descr}

{reduction_identity_dtype}

{reduction_args}

{reduction_example}"""
@_apply_docstring_templates
def argmax(input: Tensor | MaskedTensor, dim: int | None = None, *, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}
{reduction_descr}
{reduction_identity_dtype}
{reduction_args}
{reduction_example}"""
@_apply_docstring_templates
def argmin(input: Tensor | MaskedTensor, dim: int | None = None, *, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}
{reduction_descr}
{reduction_identity_dtype}
{reduction_args}
{reduction_example}"""
@_apply_docstring_templates
def mean(input: Tensor | MaskedTensor, dim: DimOrDims = None, *, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}

{reduction_descr}

By definition, the identity value of a mean operation is the mean
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
mean is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.

{reduction_args}

{reduction_example}"""
@_apply_docstring_templates
def median(input: Tensor | MaskedTensor, dim: int = -1, *, keepdim: bool = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}
{reduction_descr}
By definition, the identity value of a median operation is the median
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
median is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
@_apply_docstring_templates
def logsumexp(input: Tensor, dim: DimOrDims = None, *, keepdim: bool = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
@_apply_docstring_templates
def norm(input: Tensor | MaskedTensor, ord: float | None = 2.0, dim: DimOrDims = None, *, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}

{reduction_descr}

The identity value of norm operation, which is used to start the
reduction, is ``{identity_float32}``, except for ``ord=-inf`` it is
``{identity_ord_ninf}``.

{reduction_args}

{reduction_example}"""
@_apply_docstring_templates
def var(input: Tensor | MaskedTensor, dim: DimOrDims = None, unbiased: bool | None = None, *, correction: int | float | None = None, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}
{reduction_descr}
The identity value of sample variance operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
@_apply_docstring_templates
def std(input: Tensor | MaskedTensor, dim: DimOrDims = None, unbiased: bool | None = None, *, correction: int | None = None, keepdim: bool | None = False, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor:
    """{reduction_signature}
{reduction_descr}
The identity value of sample standard deviation operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
@_apply_docstring_templates
def softmax(input: Tensor | MaskedTensor, dim: int, *, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
@_apply_docstring_templates
def log_softmax(input: Tensor | MaskedTensor, dim: int, *, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
@_apply_docstring_templates
def softmin(input: Tensor | MaskedTensor, dim: int, *, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
@_apply_docstring_templates
def normalize(input: Tensor | MaskedTensor, ord: float, dim: int, *, eps: float = 1e-12, dtype: DType | None = None, mask: Tensor | None = None) -> Tensor: ...
