from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution

__all__ = ['InverseGamma']

class InverseGamma(TransformedDistribution):
    '''
    Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate`
    where::

        X ~ Gamma(concentration, rate)
        Y = 1 / X ~ InverseGamma(concentration, rate)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = InverseGamma(torch.tensor([2.0]), torch.tensor([3.0]))
        >>> m.sample()
        tensor([ 1.2953])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    base_dist: Gamma
    def __init__(self, concentration: Tensor | float, rate: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def concentration(self) -> Tensor: ...
    @property
    def rate(self) -> Tensor: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def entropy(self): ...
