import torch
import weakref
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

__all__ = ['AbsTransform', 'AffineTransform', 'CatTransform', 'ComposeTransform', 'CorrCholeskyTransform', 'CumulativeDistributionTransform', 'ExpTransform', 'IndependentTransform', 'LowerCholeskyTransform', 'PositiveDefiniteTransform', 'PowerTransform', 'ReshapeTransform', 'SigmoidTransform', 'SoftplusTransform', 'TanhTransform', 'SoftmaxTransform', 'StackTransform', 'StickBreakingTransform', 'Transform', 'identity_transform']

class Transform:
    """
    Abstract class for invertable transformations with computable log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Caching is useful for transforms whose inverses are either expensive or
    numerically unstable. Note that care must be taken with memoized values
    since the autograd graph may be reversed. For example while the following
    works with or without caching::

        y = t(x)
        t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.

    However the following will error when caching due to dependency reversal::

        y = t(x)
        z = t.inv(y)
        grad(z.sum(), [y])  # error because z is x

    Derived classes should implement one or both of :meth:`_call` or
    :meth:`_inverse`. Derived classes that set `bijective=True` should also
    implement :meth:`log_abs_det_jacobian`.

    Args:
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.

    Attributes:
        domain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid inputs to this transform.
        codomain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid outputs to this transform
            which are inputs to the inverse transform.
        bijective (bool): Whether this transform is bijective. A transform
            ``t`` is bijective iff ``t.inv(t(x)) == x`` and
            ``t(t.inv(y)) == y`` for every ``x`` in the domain and ``y`` in
            the codomain. Transforms that are not bijective should at least
            maintain the weaker pseudoinverse properties
            ``t(t.inv(t(x)) == t(x)`` and ``t.inv(t(t.inv(y))) == t.inv(y)``.
        sign (int or Tensor): For bijective univariate transforms, this
            should be +1 or -1 depending on whether transform is monotone
            increasing or decreasing.
    """
    bijective: bool
    domain: constraints.Constraint
    codomain: constraints.Constraint
    _cache_size: Incomplete
    _inv: weakref.ReferenceType[Transform] | None
    _cached_x_y: Incomplete
    def __init__(self, cache_size: int = 0) -> None: ...
    def __getstate__(self): ...
    @property
    def event_dim(self) -> int: ...
    @property
    def inv(self) -> Transform:
        """
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        """
    @property
    def sign(self) -> int:
        """
        Returns the sign of the determinant of the Jacobian, if applicable.
        In general this only makes sense for bijective transforms.
        """
    def with_cache(self, cache_size: int = 1): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __call__(self, x):
        """
        Computes the transform `x => y`.
        """
    def _inv_call(self, y):
        """
        Inverts the transform `y => x`.
        """
    def _call(self, x) -> None:
        """
        Abstract method to compute forward transformation.
        """
    def _inverse(self, y) -> None:
        """
        Abstract method to compute inverse transformation.
        """
    def log_abs_det_jacobian(self, x, y) -> None:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
    def __repr__(self) -> str: ...
    def forward_shape(self, shape):
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
    def inverse_shape(self, shape):
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """

class _InverseTransform(Transform):
    """
    Inverts a single :class:`Transform`.
    This class is private; please instead use the ``Transform.inv`` property.
    """
    _inv: Transform
    def __init__(self, transform: Transform) -> None: ...
    def domain(self): ...
    def codomain(self): ...
    @property
    def bijective(self) -> bool: ...
    @property
    def sign(self) -> int: ...
    @property
    def inv(self) -> Transform: ...
    def with_cache(self, cache_size: int = 1): ...
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...
    def __call__(self, x): ...
    def log_abs_det_jacobian(self, x, y): ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...

class ComposeTransform(Transform):
    """
    Composes multiple transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform`): A list of transforms to compose.
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.
    """
    parts: Incomplete
    def __init__(self, parts: list[Transform], cache_size: int = 0) -> None: ...
    def __eq__(self, other): ...
    def domain(self): ...
    def codomain(self): ...
    @lazy_property
    def bijective(self) -> bool: ...
    @lazy_property
    def sign(self) -> int: ...
    _inv: Incomplete
    @property
    def inv(self) -> Transform: ...
    def with_cache(self, cache_size: int = 1): ...
    def __call__(self, x): ...
    def log_abs_det_jacobian(self, x, y): ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...
    def __repr__(self) -> str: ...

identity_transform: Incomplete

class IndependentTransform(Transform):
    """
    Wrapper around another transform to treat
    ``reinterpreted_batch_ndims``-many extra of the right most dimensions as
    dependent. This has no effect on the forward or backward transforms, but
    does sum out ``reinterpreted_batch_ndims``-many of the rightmost dimensions
    in :meth:`log_abs_det_jacobian`.

    Args:
        base_transform (:class:`Transform`): A base transform.
        reinterpreted_batch_ndims (int): The number of extra rightmost
            dimensions to treat as dependent.
    """
    base_transform: Incomplete
    reinterpreted_batch_ndims: Incomplete
    def __init__(self, base_transform: Transform, reinterpreted_batch_ndims: int, cache_size: int = 0) -> None: ...
    def with_cache(self, cache_size: int = 1): ...
    def domain(self): ...
    def codomain(self): ...
    @property
    def bijective(self) -> bool: ...
    @property
    def sign(self) -> int: ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...
    def __repr__(self) -> str: ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...

class ReshapeTransform(Transform):
    """
    Unit Jacobian transform to reshape the rightmost part of a tensor.

    Note that ``in_shape`` and ``out_shape`` must have the same number of
    elements, just as for :meth:`torch.Tensor.reshape`.

    Arguments:
        in_shape (torch.Size): The input event shape.
        out_shape (torch.Size): The output event shape.
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported. (Default 0.)
    """
    bijective: bool
    in_shape: Incomplete
    out_shape: Incomplete
    def __init__(self, in_shape: torch.Size, out_shape: torch.Size, cache_size: int = 0) -> None: ...
    @constraints.dependent_property
    def domain(self): ...
    @constraints.dependent_property
    def codomain(self): ...
    def with_cache(self, cache_size: int = 1): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...

class ExpTransform(Transform):
    """
    Transform via the mapping :math:`y = \\exp(x)`.
    """
    domain: Incomplete
    codomain: Incomplete
    bijective: bool
    sign: int
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...

class PowerTransform(Transform):
    """
    Transform via the mapping :math:`y = x^{\\text{exponent}}`.
    """
    domain: Incomplete
    codomain: Incomplete
    bijective: bool
    def __init__(self, exponent: Tensor, cache_size: int = 0) -> None: ...
    def with_cache(self, cache_size: int = 1): ...
    @lazy_property
    def sign(self) -> int: ...
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...

class SigmoidTransform(Transform):
    """
    Transform via the mapping :math:`y = \\frac{1}{1 + \\exp(-x)}` and :math:`x = \\text{logit}(y)`.
    """
    domain: Incomplete
    codomain: Incomplete
    bijective: bool
    sign: int
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...

class SoftplusTransform(Transform):
    """
    Transform via the mapping :math:`\\text{Softplus}(x) = \\log(1 + \\exp(x))`.
    The implementation reverts to the linear function when :math:`x > 20`.
    """
    domain: Incomplete
    codomain: Incomplete
    bijective: bool
    sign: int
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...

class TanhTransform(Transform):
    """
    Transform via the mapping :math:`y = \\tanh(x)`.

    It is equivalent to

    .. code-block:: python

        ComposeTransform(
            [
                AffineTransform(0.0, 2.0),
                SigmoidTransform(),
                AffineTransform(-1.0, 2.0),
            ]
        )

    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.

    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    """
    domain: Incomplete
    codomain: Incomplete
    bijective: bool
    sign: int
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...

class AbsTransform(Transform):
    """Transform via the mapping :math:`y = |x|`."""
    domain: Incomplete
    codomain: Incomplete
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...

class AffineTransform(Transform):
    """
    Transform via the pointwise affine mapping :math:`y = \\text{loc} + \\text{scale} \\times x`.

    Args:
        loc (Tensor or float): Location parameter.
        scale (Tensor or float): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """
    bijective: bool
    loc: Incomplete
    scale: Incomplete
    _event_dim: Incomplete
    def __init__(self, loc: Tensor | float, scale: Tensor | float, event_dim: int = 0, cache_size: int = 0) -> None: ...
    @property
    def event_dim(self) -> int: ...
    def domain(self): ...
    def codomain(self): ...
    def with_cache(self, cache_size: int = 1): ...
    def __eq__(self, other): ...
    @property
    def sign(self) -> Tensor | int: ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...

class CorrCholeskyTransform(Transform):
    """
    Transforms an uncontrained real vector :math:`x` with length :math:`D*(D-1)/2` into the
    Cholesky factor of a D-dimension correlation matrix. This Cholesky factor is a lower
    triangular matrix with positive diagonals and unit Euclidean norm for each row.
    The transform is processed as follows:

        1. First we convert x into a lower triangular matrix in row order.
        2. For each row :math:`X_i` of the lower triangular part, we apply a *signed* version of
           class :class:`StickBreakingTransform` to transform :math:`X_i` into a
           unit Euclidean length vector using the following steps:
           - Scales into the interval :math:`(-1, 1)` domain: :math:`r_i = \\tanh(X_i)`.
           - Transforms into an unsigned domain: :math:`z_i = r_i^2`.
           - Applies :math:`s_i = StickBreakingTransform(z_i)`.
           - Transforms back into signed domain: :math:`y_i = sign(r_i) * \\sqrt{s_i}`.
    """
    domain: Incomplete
    codomain: Incomplete
    bijective: bool
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y, intermediates=None): ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...

class SoftmaxTransform(Transform):
    """
    Transform from unconstrained space to the simplex via :math:`y = \\exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain: Incomplete
    codomain: Incomplete
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...

class StickBreakingTransform(Transform):
    """
    Transform from unconstrained space to the simplex of one additional
    dimension via a stick-breaking process.

    This transform arises as an iterated sigmoid transform in a stick-breaking
    construction of the `Dirichlet` distribution: the first logit is
    transformed via sigmoid to the first probability and the probability of
    everything else, and then the process recurses.

    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """
    domain: Incomplete
    codomain: Incomplete
    bijective: bool
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...
    def forward_shape(self, shape): ...
    def inverse_shape(self, shape): ...

class LowerCholeskyTransform(Transform):
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries.

    This is useful for parameterizing positive definite matrices in terms of
    their Cholesky factorization.
    """
    domain: Incomplete
    codomain: Incomplete
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...

class PositiveDefiniteTransform(Transform):
    """
    Transform from unconstrained matrices to positive-definite matrices.
    """
    domain: Incomplete
    codomain: Incomplete
    def __eq__(self, other): ...
    def _call(self, x): ...
    def _inverse(self, y): ...

class CatTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`, of length `lengths[dim]`,
    in a way compatible with :func:`torch.cat`.

    Example::

       x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
       x = torch.cat([x0, x0], dim=0)
       t0 = CatTransform([ExpTransform(), identity_transform], dim=0, lengths=[10, 10])
       t = CatTransform([t0, t0], dim=0, lengths=[20, 20])
       y = t(x)
    """
    transforms: list[Transform]
    lengths: Incomplete
    dim: Incomplete
    def __init__(self, tseq: Sequence[Transform], dim: int = 0, lengths: Sequence[int] | None = None, cache_size: int = 0) -> None: ...
    @lazy_property
    def event_dim(self) -> int: ...
    @lazy_property
    def length(self) -> int: ...
    def with_cache(self, cache_size: int = 1): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...
    @property
    def bijective(self) -> bool: ...
    @constraints.dependent_property
    def domain(self): ...
    @constraints.dependent_property
    def codomain(self): ...

class StackTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`
    in a way compatible with :func:`torch.stack`.

    Example::

       x = torch.stack([torch.range(1, 10), torch.range(1, 10)], dim=1)
       t = StackTransform([ExpTransform(), identity_transform], dim=1)
       y = t(x)
    """
    transforms: list[Transform]
    dim: Incomplete
    def __init__(self, tseq: Sequence[Transform], dim: int = 0, cache_size: int = 0) -> None: ...
    def with_cache(self, cache_size: int = 1): ...
    def _slice(self, z): ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...
    @property
    def bijective(self) -> bool: ...
    @constraints.dependent_property
    def domain(self): ...
    @constraints.dependent_property
    def codomain(self): ...

class CumulativeDistributionTransform(Transform):
    """
    Transform via the cumulative distribution function of a probability distribution.

    Args:
        distribution (Distribution): Distribution whose cumulative distribution function to use for
            the transformation.

    Example::

        # Construct a Gaussian copula from a multivariate normal.
        base_dist = MultivariateNormal(
            loc=torch.zeros(2),
            scale_tril=LKJCholesky(2).sample(),
        )
        transform = CumulativeDistributionTransform(Normal(0, 1))
        copula = TransformedDistribution(base_dist, [transform])
    """
    bijective: bool
    codomain: Incomplete
    sign: int
    distribution: Incomplete
    def __init__(self, distribution: Distribution, cache_size: int = 0) -> None: ...
    @property
    def domain(self) -> constraints.Constraint | None: ...
    def _call(self, x): ...
    def _inverse(self, y): ...
    def log_abs_det_jacobian(self, x, y): ...
    def with_cache(self, cache_size: int = 1): ...
