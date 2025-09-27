import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.types import _size

__all__ = ['OneHotCategorical', 'OneHotCategoricalStraightThrough']

class OneHotCategorical(Distribution):
    '''
    Creates a one-hot categorical distribution parameterized by :attr:`probs` or
    :attr:`logits`.

    Samples are one-hot coded vectors of size ``probs.size(-1)``.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.distributions.Categorical` for specifications of
    :attr:`probs` and :attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 0.,  0.,  0.,  1.])

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_enumerate_support: bool
    _categorical: Incomplete
    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def _new(self, *args, **kwargs): ...
    @property
    def _param(self) -> Tensor: ...
    @property
    def probs(self) -> Tensor: ...
    @property
    def logits(self) -> Tensor: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    @property
    def param_shape(self) -> torch.Size: ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def entropy(self): ...
    def enumerate_support(self, expand: bool = True): ...

class OneHotCategoricalStraightThrough(OneHotCategorical):
    """
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al., 2013)
    """
    has_rsample: bool
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
