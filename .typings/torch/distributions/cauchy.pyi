from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.types import _size

__all__ = ['Cauchy']

class Cauchy(Distribution):
    '''
    Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
    independent normally distributed random variables with means `0` follows a
    Cauchy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
        tensor([ 2.3214])

    Args:
        loc (float or Tensor): mode or median of the distribution.
        scale (float or Tensor): half width at half maximum.
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...
    def cdf(self, value): ...
    def icdf(self, value): ...
    def entropy(self): ...
