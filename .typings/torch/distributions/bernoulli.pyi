import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import lazy_property
from torch.types import Number

__all__ = ['Bernoulli']

class Bernoulli(ExponentialFamily):
    '''
    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Bernoulli(torch.tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
        validate_args (bool, optional): whether to validate arguments, None by default
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_enumerate_support: bool
    _mean_carrier_measure: int
    _param: Incomplete
    def __init__(self, probs: Tensor | Number | None = None, logits: Tensor | Number | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def _new(self, *args, **kwargs): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    @lazy_property
    def logits(self) -> Tensor: ...
    @lazy_property
    def probs(self) -> Tensor: ...
    @property
    def param_shape(self) -> torch.Size: ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def entropy(self): ...
    def enumerate_support(self, expand: bool = True): ...
    @property
    def _natural_params(self) -> tuple[Tensor]: ...
    def _log_normalizer(self, x): ...
