from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.types import _size

__all__ = ['Beta']

class Beta(ExponentialFamily):
    '''
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
        tensor([ 0.1046])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    _dirichlet: Incomplete
    def __init__(self, concentration1: Tensor | float, concentration0: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def rsample(self, sample_shape: _size = ()) -> Tensor: ...
    def log_prob(self, value): ...
    def entropy(self): ...
    @property
    def concentration1(self) -> Tensor: ...
    @property
    def concentration0(self) -> Tensor: ...
    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]: ...
    def _log_normalizer(self, x, y): ...
