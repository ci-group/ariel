from torch import Tensor
from torch.types import Number
from typing import Any, Callable, Generic, TypeVar, overload

__all__ = ['broadcast_all', 'logits_to_probs', 'clamp_probs', 'probs_to_logits', 'lazy_property', 'tril_matrix_to_vec', 'vec_to_tril_matrix']

def broadcast_all(*values: Tensor | Number) -> tuple[Tensor, ...]:
    """
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `Number`, `torch.*Tensor` or objects implementing __torch_function__)

    Raises:
        ValueError: if any of the values is not a `Number` instance,
            a `torch.*Tensor` instance, or an instance implementing __torch_function__
    """
def logits_to_probs(logits: Tensor, is_binary: bool = False) -> Tensor:
    """
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
def clamp_probs(probs: Tensor) -> Tensor:
    """Clamps the probabilities to be in the open interval `(0, 1)`.

    The probabilities would be clamped between `eps` and `1 - eps`,
    and `eps` would be the smallest representable positive number for the input data type.

    Args:
        probs (Tensor): A tensor of probabilities.

    Returns:
        Tensor: The clamped probabilities.

    Examples:
        >>> probs = torch.tensor([0.0, 0.5, 1.0])
        >>> clamp_probs(probs)
        tensor([1.1921e-07, 5.0000e-01, 1.0000e+00])

        >>> probs = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        >>> clamp_probs(probs)
        tensor([2.2204e-16, 5.0000e-01, 1.0000e+00], dtype=torch.float64)

    """
def probs_to_logits(probs: Tensor, is_binary: bool = False) -> Tensor:
    """
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
T = TypeVar('T', contravariant=True)
R = TypeVar('R', covariant=True)

class lazy_property(Generic[T, R]):
    """
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    wrapped: Callable[[T], R]
    def __init__(self, wrapped: Callable[[T], R]) -> None: ...
    @overload
    def __get__(self, instance: None, obj_type: Any = None) -> _lazy_property_and_property[T, R]: ...
    @overload
    def __get__(self, instance: T, obj_type: Any = None) -> R: ...

class _lazy_property_and_property(lazy_property[T, R], property):
    """We want lazy properties to look like multiple things.

    * property when Sphinx autodoc looks
    * lazy_property when Distribution validate_args looks
    """
    def __init__(self, wrapped: Callable[[T], R]) -> None: ...

def tril_matrix_to_vec(mat: Tensor, diag: int = 0) -> Tensor:
    """
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
def vec_to_tril_matrix(vec: Tensor, diag: int = 0) -> Tensor:
    """
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
