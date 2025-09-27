from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution

__all__ = ['HalfCauchy']

class HalfCauchy(TransformedDistribution):
    '''
    Creates a half-Cauchy distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    base_dist: Cauchy
    def __init__(self, scale: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def scale(self) -> Tensor: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def log_prob(self, value): ...
    def cdf(self, value): ...
    def icdf(self, prob): ...
    def entropy(self): ...
