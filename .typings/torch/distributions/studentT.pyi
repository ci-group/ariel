from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.types import _size

__all__ = ['StudentT']

class StudentT(Distribution):
    '''
    Creates a Student\'s t-distribution parameterized by degree of
    freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = StudentT(torch.tensor([2.0]))
        >>> m.sample()  # Student\'s t-distributed with degrees of freedom=2
        tensor([ 0.1046])

    Args:
        df (float or Tensor): degrees of freedom
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    _chi2: Incomplete
    def __init__(self, df: Tensor | float, loc: Tensor | float = 0.0, scale: Tensor | float = 1.0, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...
    def entropy(self): ...
