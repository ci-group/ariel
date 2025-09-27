from _typeshed import Incomplete
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import Transform
from torch.types import _size

__all__ = ['TransformedDistribution']

class TransformedDistribution(Distribution):
    """
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
        log p(Y) = log p(X) + log |det (dX/dY)|

    Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
    maximum shape of its base distribution and its transforms, since transforms
    can introduce correlations among events.

    An example for the usage of :class:`TransformedDistribution` would be::

        # Building a Logistic Distribution
        # X ~ Uniform(0, 1)
        # f = a + b * logit(X)
        # Y ~ f(X) ~ Logistic(a, b)
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
        logistic = TransformedDistribution(base_distribution, transforms)

    For more examples, please look at the implementations of
    :class:`~torch.distributions.gumbel.Gumbel`,
    :class:`~torch.distributions.half_cauchy.HalfCauchy`,
    :class:`~torch.distributions.half_normal.HalfNormal`,
    :class:`~torch.distributions.log_normal.LogNormal`,
    :class:`~torch.distributions.pareto.Pareto`,
    :class:`~torch.distributions.weibull.Weibull`,
    :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli` and
    :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`
    """
    arg_constraints: dict[str, constraints.Constraint]
    transforms: Incomplete
    base_dist: Incomplete
    def __init__(self, base_distribution: Distribution, transforms: Transform | list[Transform], validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def support(self): ...
    @property
    def has_rsample(self) -> bool: ...
    def sample(self, sample_shape=...):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
    def rsample(self, sample_shape: _size = ...) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
    def _monotonize_cdf(self, value):
        """
        This conditionally flips ``value -> 1-value`` to ensure :meth:`cdf` is
        monotone increasing.
        """
    def cdf(self, value):
        """
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        """
    def icdf(self, value):
        """
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        """
