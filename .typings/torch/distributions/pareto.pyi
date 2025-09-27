from _typeshed import Incomplete
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.types import _size

__all__ = ['Pareto']

class Pareto(TransformedDistribution):
    '''
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    '''
    arg_constraints: Incomplete
    def __init__(self, scale: Tensor | float, alpha: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape: _size, _instance: Pareto | None = None) -> Pareto: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def support(self) -> constraints.Constraint: ...
    def entropy(self) -> Tensor: ...
