from _typeshed import Incomplete
from torch import Tensor
from torch.distributions import Categorical, constraints
from torch.distributions.distribution import Distribution

__all__ = ['MixtureSameFamily']

class MixtureSameFamily(Distribution):
    '''
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all component are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` component) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.

    Examples::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        >>> # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)

        >>> # Construct Gaussian Mixture Model in 2D consisting of 5 equally
        >>> # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
        ...          torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

        >>> # Construct a batch of 3 Gaussian Mixture Models in 2D each
        >>> # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
        ...         torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting component.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    '''
    arg_constraints: dict[str, constraints.Constraint]
    has_rsample: bool
    _mixture_distribution: Incomplete
    _component_distribution: Incomplete
    _num_component: Incomplete
    _event_ndims: Incomplete
    def __init__(self, mixture_distribution: Categorical, component_distribution: Distribution, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @constraints.dependent_property
    def support(self): ...
    @property
    def mixture_distribution(self) -> Categorical: ...
    @property
    def component_distribution(self) -> Distribution: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def cdf(self, x): ...
    def log_prob(self, x): ...
    def sample(self, sample_shape=...): ...
    def _pad(self, x): ...
    def _pad_mixture_dimensions(self, x): ...
    def __repr__(self) -> str: ...
