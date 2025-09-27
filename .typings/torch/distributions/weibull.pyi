from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.transformed_distribution import TransformedDistribution

__all__ = ['Weibull']

class Weibull(TransformedDistribution):
    '''
    Samples from a two-parameter Weibull distribution.

    Example:

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
        tensor([ 0.4784])

    Args:
        scale (float or Tensor): Scale parameter of distribution (lambda).
        concentration (float or Tensor): Concentration parameter of distribution (k/shape).
        validate_args (bool, optional): Whether to validate arguments. Default: None.
    '''
    arg_constraints: Incomplete
    support: Incomplete
    concentration_reciprocal: Incomplete
    def __init__(self, scale: Tensor | float, concentration: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def entropy(self): ...
