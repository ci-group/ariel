import torch
from .grad_mode import _DecoratorContextManager
from _typeshed import Incomplete
from typing import Any, NamedTuple

__all__ = ['UnpackedDualTensor', 'enter_dual_level', 'exit_dual_level', 'make_dual', 'unpack_dual', 'dual_level']

def enter_dual_level():
    """Enter a new forward grad level.

    This level can be used to make and unpack dual Tensors to compute
    forward gradients.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
def exit_dual_level(*, level=None) -> None:
    """Exit a forward grad level.

    This function deletes all the gradients associated with this
    level. Only deleting the latest entered level is allowed.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
def make_dual(tensor, tangent, *, level=None):
    '''Associate a tensor value with its tangent to create a "dual tensor" for forward AD gradient computation.

    The result is a new tensor aliased to :attr:`tensor` with :attr:`tangent` embedded
    as an attribute as-is if it has the same storage layout or copied otherwise.
    The tangent attribute can be recovered with :func:`unpack_dual`.

    This function is backward differentiable.

    Given a function `f` whose jacobian is `J`, it allows one to compute the Jacobian-vector product (`jvp`)
    between `J` and a given vector `v` as follows.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, v)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.

    '''

class UnpackedDualTensor(NamedTuple):
    """Namedtuple returned by :func:`unpack_dual` containing the primal and tangent components of the dual tensor.

    See :func:`unpack_dual` for more details.
    """
    primal: torch.Tensor
    tangent: torch.Tensor | None

def unpack_dual(tensor, *, level=None):
    '''Unpack a "dual tensor" to get both its Tensor value and its forward AD gradient.

    The result is a namedtuple ``(primal, tangent)`` where ``primal`` is a view of
    :attr:`tensor`\'s primal and ``tangent`` is :attr:`tensor`\'s tangent as-is.
    Neither of these tensors can be dual tensor of level :attr:`level`.

    This function is backward differentiable.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)
        ...     jvp = unpack_dual(out).tangent

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    '''

class dual_level(_DecoratorContextManager):
    '''Context-manager for forward AD, where all forward AD computation must occur within the ``dual_level`` context.

    .. Note::

        The ``dual_level`` context appropriately enters and exit the dual level to
        controls the current forward AD level, which is used by default by the other
        functions in this API.

        We currently don\'t plan to support nested ``dual_level`` contexts, however, so
        only a single forward AD level is supported. To compute higher-order
        forward grads, one can use :func:`torch.func.jvp`.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> x = torch.tensor([1])
        >>> x_t = torch.tensor([1])
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     # Do computations with inp
        ...     out = your_fn(inp)
        ...     _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    '''
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
_is_fwd_grad_enabled = torch._C._is_fwd_grad_enabled

class _set_fwd_grad_enabled(_DecoratorContextManager):
    prev: Incomplete
    def __init__(self, mode: bool) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
