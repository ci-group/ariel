from _typeshed import Incomplete
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.types import _size
from typing import Generic, TypeVar

__all__ = ['Independent']

D = TypeVar('D', bound=Distribution)

class Independent(Distribution, Generic[D]):
    """
    Reinterprets some of the batch dims of a distribution as event dims.

    This is mainly useful for changing the shape of the result of
    :meth:`log_prob`. For example to create a diagonal Normal distribution with
    the same shape as a Multivariate Normal distribution (so they are
    interchangeable), you can::

        >>> from torch.distributions.multivariate_normal import MultivariateNormal
        >>> from torch.distributions.normal import Normal
        >>> loc = torch.zeros(3)
        >>> scale = torch.ones(3)
        >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        >>> [mvn.batch_shape, mvn.event_shape]
        [torch.Size([]), torch.Size([3])]
        >>> normal = Normal(loc, scale)
        >>> [normal.batch_shape, normal.event_shape]
        [torch.Size([3]), torch.Size([])]
        >>> diagn = Independent(normal, 1)
        >>> [diagn.batch_shape, diagn.event_shape]
        [torch.Size([]), torch.Size([3])]

    Args:
        base_distribution (torch.distributions.distribution.Distribution): a
            base distribution
        reinterpreted_batch_ndims (int): the number of batch dims to
            reinterpret as event dims
    """
    arg_constraints: dict[str, constraints.Constraint]
    base_dist: D
    reinterpreted_batch_ndims: Incomplete
    def __init__(self, base_distribution: D, reinterpreted_batch_ndims: int, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def has_rsample(self) -> bool: ...
    @property
    def has_enumerate_support(self) -> bool: ...
    @constraints.dependent_property
    def support(self): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def sample(self, sample_shape=...) -> Tensor: ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...
    def entropy(self): ...
    def enumerate_support(self, expand: bool = True): ...
    def __repr__(self) -> str: ...
