from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property
from torch.types import Number

__all__ = ['Geometric']

class Geometric(Distribution):
    '''
    Creates a Geometric distribution parameterized by :attr:`probs`,
    where :attr:`probs` is the probability of success of Bernoulli trials.

    .. math::

        P(X=k) = (1-p)^{k} p, k = 0, 1, ...

    .. note::
        :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th trial is the first success
        hence draws samples in :math:`\\{0, 1, \\ldots\\}`, whereas
        :func:`torch.Tensor.geometric_` `k`-th trial is the first success hence draws samples in :math:`\\{1, 2, \\ldots\\}`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Geometric(torch.tensor([0.3]))
        >>> m.sample()  # underlying Bernoulli has 30% chance 1; 70% chance 0
        tensor([ 2.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`. Must be in range (0, 1]
        logits (Number, Tensor): the log-odds of sampling `1`.
    '''
    arg_constraints: Incomplete
    support: Incomplete
    def __init__(self, probs: Tensor | Number | None = None, logits: Tensor | Number | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
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
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def entropy(self): ...
