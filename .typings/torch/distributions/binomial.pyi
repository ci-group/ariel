import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

__all__ = ['Binomial']

class Binomial(Distribution):
    '''
    Creates a Binomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be
    broadcastable with :attr:`probs`/:attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
        tensor([   0.,   22.,   71.,  100.])

        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
        >>> x = m.sample()
        tensor([[ 4.,  5.],
                [ 7.,  6.]])

    Args:
        total_count (int or Tensor): number of Bernoulli trials
        probs (Tensor): Event probabilities
        logits (Tensor): Event log-odds
    '''
    arg_constraints: Incomplete
    has_enumerate_support: bool
    total_count: Incomplete
    _param: Incomplete
    def __init__(self, total_count: Tensor | int = 1, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def _new(self, *args, **kwargs): ...
    def support(self): ...
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
