from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.types import _size

__all__ = ['Normal']

class Normal(ExponentialFamily):
    '''
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    _mean_carrier_measure: int
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def stddev(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def sample(self, sample_shape=...): ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...
    def cdf(self, value): ...
    def icdf(self, value): ...
    def entropy(self): ...
    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]: ...
    def _log_normalizer(self, x, y): ...
