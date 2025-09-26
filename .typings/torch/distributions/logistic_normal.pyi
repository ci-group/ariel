from _typeshed import Incomplete
from torch import Tensor
from torch.distributions import Independent
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution

__all__ = ['LogisticNormal']

class LogisticNormal(TransformedDistribution):
    '''
    Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the base `Normal` distribution transformed with the
    `StickBreakingTransform` such that::

        X ~ LogisticNormal(loc, scale)
        Y = log(X / (1 - X.cumsum(-1)))[..., :-1] ~ Normal(loc, scale)

    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution

    Example::

        >>> # logistic-normal distributed with mean=(0, 0, 0) and stddev=(1, 1, 1)
        >>> # of the base Normal distribution
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogisticNormal(torch.tensor([0.0] * 3), torch.tensor([1.0] * 3))
        >>> m.sample()
        tensor([ 0.7653,  0.0341,  0.0579,  0.1427])

    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    base_dist: Independent[Normal]
    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def loc(self) -> Tensor: ...
    @property
    def scale(self) -> Tensor: ...
