import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

__all__ = ['Categorical']

class Categorical(Distribution):
    '''
    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from :math:`\\{0, \\ldots, K-1\\}` where `K` is ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.multinomial`

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3)

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    '''
    arg_constraints: Incomplete
    has_enumerate_support: bool
    _param: Incomplete
    _num_events: Incomplete
    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def _new(self, *args, **kwargs): ...
    def support(self): ...
    @lazy_property
    def logits(self) -> Tensor: ...
    @lazy_property
    def probs(self) -> Tensor: ...
    @property
    def param_shape(self) -> torch.Size: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def entropy(self): ...
    def enumerate_support(self, expand: bool = True): ...
