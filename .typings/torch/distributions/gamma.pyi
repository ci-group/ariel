from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.types import _size

__all__ = ['Gamma']

class Gamma(ExponentialFamily):
    '''
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate parameter of the distribution
            (often referred to as beta), rate = 1 / scale
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    _mean_carrier_measure: int
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def __init__(self, concentration: Tensor | float, rate: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...
    def entropy(self): ...
    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]: ...
    def _log_normalizer(self, x, y): ...
    def cdf(self, value): ...
