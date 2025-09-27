import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution

__all__ = ['Multinomial']

class Multinomial(Distribution):
    '''
    Creates a Multinomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). The innermost dimension of
    :attr:`probs` indexes over categories. All other dimensions index over batches.

    Note that :attr:`total_count` need not be specified if only :meth:`log_prob` is
    called (see example below)

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    -   :meth:`sample` requires a single shared `total_count` for all
        parameters and samples.
    -   :meth:`log_prob` allows different `total_count` for each parameter and
        sample.

    Example::

        >>> # xdoctest: +SKIP("FIXME: found invalid values")
        >>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
        >>> x = m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 21.,  24.,  30.,  25.])

        >>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
        tensor([-4.1338])

    Args:
        total_count (int): number of trials
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    '''
    arg_constraints: Incomplete
    total_count: int
    @property
    def mean(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    _categorical: Incomplete
    _binomial: Incomplete
    def __init__(self, total_count: int = 1, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def _new(self, *args, **kwargs): ...
    def support(self): ...
    @property
    def logits(self) -> Tensor: ...
    @property
    def probs(self) -> Tensor: ...
    @property
    def param_shape(self) -> torch.Size: ...
    def sample(self, sample_shape=...): ...
    def entropy(self): ...
    def log_prob(self, value): ...
