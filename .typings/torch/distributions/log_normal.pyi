from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution

__all__ = ['LogNormal']

class LogNormal(TransformedDistribution):
    '''
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    base_dist: Normal
    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def loc(self) -> Tensor: ...
    @property
    def scale(self) -> Tensor: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def entropy(self): ...
