import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import lazy_property
from torch.types import Number, _size

__all__ = ['ContinuousBernoulli']

class ContinuousBernoulli(ExponentialFamily):
    '''
    Creates a continuous Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    The distribution is supported in [0, 1] and parameterized by \'probs\' (in
    (0,1)) or \'logits\' (real-valued). Note that, unlike the Bernoulli, \'probs\'
    does not correspond to a probability and \'logits\' does not correspond to
    log-odds, but the same names are used due to the similarity with the
    Bernoulli. See [1] for more details.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = ContinuousBernoulli(torch.tensor([0.3]))
        >>> m.sample()
        tensor([ 0.2538])

    Args:
        probs (Number, Tensor): (0,1) valued parameters
        logits (Number, Tensor): real valued parameters whose sigmoid matches \'probs\'

    [1] The continuous Bernoulli: fixing a pervasive error in variational
    autoencoders, Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    https://arxiv.org/abs/1907.06845
    '''
    arg_constraints: Incomplete
    support: Incomplete
    _mean_carrier_measure: int
    has_rsample: bool
    _param: Incomplete
    _lims: Incomplete
    def __init__(self, probs: Tensor | Number | None = None, logits: Tensor | Number | None = None, lims: tuple[float, float] = (0.499, 0.501), validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def _new(self, *args, **kwargs): ...
    def _outside_unstable_region(self): ...
    def _cut_probs(self): ...
    def _cont_bern_log_norm(self):
        """computes the log normalizing constant as a function of the 'probs' parameter"""
    @property
    def mean(self) -> Tensor: ...
    @property
    def stddev(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    @lazy_property
    def logits(self) -> Tensor: ...
    @lazy_property
    def probs(self) -> Tensor: ...
    @property
    def param_shape(self) -> torch.Size: ...
    def sample(self, sample_shape=...): ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...
    def cdf(self, value): ...
    def icdf(self, value): ...
    def entropy(self): ...
    @property
    def _natural_params(self) -> tuple[Tensor]: ...
    def _log_normalizer(self, x):
        """computes the log normalizing constant as a function of the natural parameter"""
