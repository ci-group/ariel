import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.types import _size

__all__ = ['ExpRelaxedCategorical', 'RelaxedOneHotCategorical']

class ExpRelaxedCategorical(Distribution):
    """
    Creates a ExpRelaxedCategorical parameterized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both).
    Returns the log of a point in the simplex. Based on the interface to
    :class:`OneHotCategorical`.

    Implementation based on [1].

    See also: :func:`torch.distributions.OneHotCategorical`

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al., 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al., 2017)
    """
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    _categorical: Incomplete
    temperature: Incomplete
    def __init__(self, temperature: Tensor, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def _new(self, *args, **kwargs): ...
    @property
    def param_shape(self) -> torch.Size: ...
    @property
    def logits(self) -> Tensor: ...
    @property
    def probs(self) -> Tensor: ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...

class RelaxedOneHotCategorical(TransformedDistribution):
    '''
    Creates a RelaxedOneHotCategorical distribution parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
    This is a relaxed version of the :class:`OneHotCategorical` distribution, so
    its samples are on simplex, and are reparametrizable.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
        ...                              torch.tensor([0.1, 0.2, 0.3, 0.4]))
        >>> m.sample()
        tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    base_dist: ExpRelaxedCategorical
    def __init__(self, temperature: Tensor, probs: Tensor | None = None, logits: Tensor | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def temperature(self) -> Tensor: ...
    @property
    def logits(self) -> Tensor: ...
    @property
    def probs(self) -> Tensor: ...
