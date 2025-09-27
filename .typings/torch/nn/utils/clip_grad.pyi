import torch
import typing
from torch import Tensor as Tensor
from torch.utils._foreach_utils import _device_has_foreach_support as _device_has_foreach_support, _group_tensors_by_device_and_dtype as _group_tensors_by_device_and_dtype, _has_foreach_support as _has_foreach_support

__all__: list[str]
_tensor_or_tensors = torch.Tensor | typing.Iterable[torch.Tensor]

def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """
@_no_grad
def _get_total_norm(tensors: _tensor_or_tensors, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None) -> torch.Tensor:
    """Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
@_no_grad
def _clip_grads_with_norm_(parameters: _tensor_or_tensors, max_norm: float, total_norm: torch.Tensor, foreach: bool | None = None) -> None:
    """Scale the gradients of an iterable of parameters given a pre-calculated total norm and desired max norm.

    The gradients will be scaled by the following calculation

    .. math::
        grad = grad * \\frac{max\\_norm}{total\\_norm + 1e-6}

    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.clip_grad_norm_` with a pre-calculated
    total norm.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        total_norm (Tensor): total norm of the gradients to use for clipping
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        None
    """
@_no_grad
def clip_grad_norm_(parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None) -> torch.Tensor:
    """Clip the gradient norm of an iterable of parameters.

    The norm is computed over the norms of the individual gradients of all parameters,
    as if the norms of the individual gradients were concatenated into a single vector.
    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.get_total_norm` followed by
    :func:`torch.nn.utils.clip_grads_with_norm_` with the ``total_norm`` returned by ``get_total_norm``.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float, optional): type of the used p-norm. Can be ``'inf'`` for
            infinity norm. Default: 2.0
        error_if_nonfinite (bool, optional): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False
        foreach (bool, optional): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
def clip_grad_norm(parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None) -> torch.Tensor:
    """Clip the gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    """
@_no_grad
def clip_grad_value_(parameters: _tensor_or_tensors, clip_value: float, foreach: bool | None = None) -> None:
    """Clip the gradients of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\\left[\\text{-clip\\_value}, \\text{clip\\_value}\\right]`
        foreach (bool, optional): use the faster foreach-based implementation
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and
            silently fall back to the slow implementation for other device types.
            Default: ``None``
    """
