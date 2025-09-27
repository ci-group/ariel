from _typeshed import Incomplete
from torch.distributions import Distribution

__all__ = ['GeneralizedPareto']

class GeneralizedPareto(Distribution):
    '''
    Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.

    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.
    Special cases include Exponential (when :attr:`loc` = 0, :attr:`concentration` = 0), Pareto (when :attr:`concentration` > 0,
    :attr:`loc` = :attr:`scale` / :attr:`concentration`), and Uniform (when :attr:`concentration` = -1).

    This distribution is often used to model the tails of other distributions. This implementation is based on the
    implementation in TensorFlow Probability.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = GeneralizedPareto(torch.tensor([0.1]), torch.tensor([2.0]), torch.tensor([0.4]))
        >>> m.sample()  # sample from a Generalized Pareto distribution with loc=0.1, scale=2.0, and concentration=0.4
        tensor([ 1.5623])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
        concentration (float or Tensor): Concentration parameter of the distribution
    '''
    arg_constraints: Incomplete
    has_rsample: bool
    def __init__(self, loc, scale, concentration, validate_args=None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def rsample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def log_survival_function(self, value): ...
    def log_cdf(self, value): ...
    def cdf(self, value): ...
    def icdf(self, value): ...
    def _z(self, x): ...
    @property
    def mean(self): ...
    @property
    def variance(self): ...
    def entropy(self): ...
    @property
    def mode(self): ...
    def support(self): ...
