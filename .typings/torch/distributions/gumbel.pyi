from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.transformed_distribution import TransformedDistribution

__all__ = ['Gumbel']

class Gumbel(TransformedDistribution):
    '''
    Samples from a Gumbel Distribution.

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
        tensor([ 1.0124])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    '''
    arg_constraints: Incomplete
    support: Incomplete
    def __init__(self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def log_prob(self, value): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def stddev(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def entropy(self): ...
