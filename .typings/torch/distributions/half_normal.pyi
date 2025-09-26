from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution

__all__ = ['HalfNormal']

class HalfNormal(TransformedDistribution):
    '''
    Creates a half-normal distribution parameterized by `scale` where::

        X ~ Normal(0, scale)
        Y = |X| ~ HalfNormal(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfNormal(torch.tensor([1.0]))
        >>> m.sample()  # half-normal distributed with scale=1
        tensor([ 0.1046])

    Args:
        scale (float or Tensor): scale of the full Normal distribution
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    base_dist: Normal
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
