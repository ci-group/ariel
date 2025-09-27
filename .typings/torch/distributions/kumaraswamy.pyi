from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.transformed_distribution import TransformedDistribution

__all__ = ['Kumaraswamy']

class Kumaraswamy(TransformedDistribution):
    '''
    Samples from a Kumaraswamy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution with concentration alpha=1 and beta=1
        tensor([ 0.1729])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    def __init__(self, concentration1: Tensor | float, concentration0: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def entropy(self): ...
