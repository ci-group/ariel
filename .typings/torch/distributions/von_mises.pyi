import torch.jit
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

__all__ = ['VonMises']

class VonMises(Distribution):
    '''
    A circular von Mises distribution.

    This implementation uses polar coordinates. The ``loc`` and ``value`` args
    can be any real number (to facilitate unconstrained optimization), but are
    interpreted as angles modulo 2 pi.

    Example::
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = VonMises(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # von Mises distributed with loc=1 and concentration=1
        tensor([1.9777])

    :param torch.Tensor loc: an angle in radians.
    :param torch.Tensor concentration: concentration parameter
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    def __init__(self, loc: Tensor, concentration: Tensor, validate_args: bool | None = None) -> None: ...
    def log_prob(self, value): ...
    @lazy_property
    def _loc(self) -> Tensor: ...
    @lazy_property
    def _concentration(self) -> Tensor: ...
    @lazy_property
    def _proposal_r(self) -> Tensor: ...
    def sample(self, sample_shape=...):
        '''
        The sampling algorithm for the von Mises distribution is based on the
        following paper: D.J. Best and N.I. Fisher, "Efficient simulation of the
        von Mises distribution." Applied Statistics (1979): 152-157.

        Sampling is always done in double precision internally to avoid a hang
        in _rejection_sample() for small values of the concentration, which
        starts to happen for single precision around 1e-4 (see issue #88443).
        '''
    def expand(self, batch_shape, _instance=None): ...
    @property
    def mean(self) -> Tensor:
        """
        The provided mean is the circular one.
        """
    @property
    def mode(self) -> Tensor: ...
    @lazy_property
    def variance(self) -> Tensor:
        """
        The provided variance is the circular one.
        """
