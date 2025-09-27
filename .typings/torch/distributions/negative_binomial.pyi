import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.utils import lazy_property

__all__ = ['NegativeBinomial']

class NegativeBinomial(Distribution):
    """
    Creates a Negative Binomial distribution, i.e. distribution
    of the number of successful independent and identical Bernoulli trials
    before :attr:`total_count` failures are achieved. The probability
    of success of each Bernoulli trial is :attr:`probs`.

    Args:
        total_count (float or Tensor): non-negative number of negative Bernoulli
            trials to stop, although the distribution is still valid for real
            valued count
        probs (Tensor): Event probabilities of success in the half open interval [0, 1)
        logits (Tensor): Event log-odds for probabilities of success
    """
    arg_constraints: Incomplete
    support: Incomplete
    total_count: Incomplete
    _param: Incomplete
    def __init__(self, total_count: Tensor | float, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None: ...
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
    @lazy_property
    def _gamma(self) -> Gamma: ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
