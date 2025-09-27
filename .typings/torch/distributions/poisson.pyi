from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.types import Number

__all__ = ['Poisson']

class Poisson(ExponentialFamily):
    '''
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \\mathrm{rate}^k \\frac{e^{-\\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for \'Long\'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    '''
    arg_constraints: Incomplete
    support: Incomplete
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def __init__(self, rate: Tensor | Number, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    @property
    def _natural_params(self) -> tuple[Tensor]: ...
    def _log_normalizer(self, x): ...
