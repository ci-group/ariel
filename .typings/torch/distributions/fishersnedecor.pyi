from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.types import _size

__all__ = ['FisherSnedecor']

class FisherSnedecor(Distribution):
    '''
    Creates a Fisher-Snedecor distribution parameterized by :attr:`df1` and :attr:`df2`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
        tensor([ 0.2453])

    Args:
        df1 (float or Tensor): degrees of freedom parameter 1
        df2 (float or Tensor): degrees of freedom parameter 2
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    _gamma1: Incomplete
    _gamma2: Incomplete
    def __init__(self, df1: Tensor | float, df2: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...
