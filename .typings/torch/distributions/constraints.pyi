from _typeshed import Incomplete
from typing import Any, Callable

__all__ = ['Constraint', 'boolean', 'cat', 'corr_cholesky', 'dependent', 'dependent_property', 'greater_than', 'greater_than_eq', 'independent', 'integer_interval', 'interval', 'half_open_interval', 'is_dependent', 'less_than', 'lower_cholesky', 'lower_triangular', 'MixtureSameFamilyConstraint', 'multinomial', 'nonnegative', 'nonnegative_integer', 'one_hot', 'positive', 'positive_semidefinite', 'positive_definite', 'positive_integer', 'real', 'real_vector', 'simplex', 'square', 'stack', 'symmetric', 'unit_interval']

class Constraint:
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.

    Attributes:
        is_discrete (bool): Whether constrained space is discrete.
            Defaults to False.
        event_dim (int): Number of rightmost dimensions that together define
            an event. The :meth:`check` method will remove this many dimensions
            when computing validity.
    """
    is_discrete: bool
    event_dim: int
    def check(self, value) -> None:
        """
        Returns a byte tensor of ``sample_shape + batch_shape`` indicating
        whether each event in value satisfies this constraint.
        """
    def __repr__(self) -> str: ...

class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.

    Args:
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """
    _is_discrete: Incomplete
    _event_dim: Incomplete
    def __init__(self, *, is_discrete=..., event_dim=...) -> None: ...
    @property
    def is_discrete(self) -> bool: ...
    @property
    def event_dim(self) -> int: ...
    def __call__(self, *, is_discrete=..., event_dim=...):
        """
        Support for syntax to customize static attributes::

            constraints.dependent(is_discrete=True, event_dim=1)
        """
    def check(self, x) -> None: ...

def is_dependent(constraint):
    '''
    Checks if ``constraint`` is a ``_Dependent`` object.

    Args:
        constraint : A ``Constraint`` object.

    Returns:
        ``bool``: True if ``constraint`` can be refined to the type ``_Dependent``, False otherwise.

    Examples:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch.distributions.constraints import is_dependent

        >>> dist = Bernoulli(probs=torch.tensor([0.6], requires_grad=True))
        >>> constraint1 = dist.arg_constraints["probs"]
        >>> constraint2 = dist.arg_constraints["logits"]

        >>> for constraint in [constraint1, constraint2]:
        >>>     if is_dependent(constraint):
        >>>         continue
    '''

class _DependentProperty(property, _Dependent):
    """
    Decorator that extends @property to act like a `Dependent` constraint when
    called on a class and act like a property when called on an object.

    Example::

        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high

            @constraints.dependent_property(is_discrete=False, event_dim=0)
            def support(self):
                return constraints.interval(self.low, self.high)

    Args:
        fn (Callable): The function to be decorated.
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """
    _is_discrete: Incomplete
    _event_dim: Incomplete
    def __init__(self, fn: Callable[..., Any] | None = None, *, is_discrete: bool | None = ..., event_dim: int | None = ...) -> None: ...
    def __call__(self, fn: Callable[..., Any]) -> _DependentProperty:
        """
        Support for syntax to customize static attributes::

            @constraints.dependent_property(is_discrete=True, event_dim=1)
            def support(self): ...
        """

class _IndependentConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """
    base_constraint: Incomplete
    reinterpreted_batch_ndims: Incomplete
    def __init__(self, base_constraint, reinterpreted_batch_ndims) -> None: ...
    @property
    def is_discrete(self) -> bool: ...
    @property
    def event_dim(self) -> int: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class MixtureSameFamilyConstraint(Constraint):
    """
    Constraint for the :class:`~torch.distribution.MixtureSameFamily`
    distribution that adds back the rightmost batch dimension before
    performing the validity check with the component distribution
    constraint.

    Args:
        base_constraint: The ``Constraint`` object of
            the component distribution of
            the :class:`~torch.distribution.MixtureSameFamily` distribution.
    """
    base_constraint: Incomplete
    def __init__(self, base_constraint) -> None: ...
    @property
    def is_discrete(self) -> bool: ...
    @property
    def event_dim(self) -> int: ...
    def check(self, value):
        """
        Check validity of ``value`` as a possible outcome of sampling
        the :class:`~torch.distribution.MixtureSameFamily` distribution.
        """
    def __repr__(self) -> str: ...

class _Boolean(Constraint):
    """
    Constrain to the two values `{0, 1}`.
    """
    is_discrete: bool
    def check(self, value): ...

class _OneHot(Constraint):
    """
    Constrain to one-hot vectors.
    """
    is_discrete: bool
    event_dim: int
    def check(self, value): ...

class _IntegerInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound]`.
    """
    is_discrete: bool
    lower_bound: Incomplete
    upper_bound: Incomplete
    def __init__(self, lower_bound, upper_bound) -> None: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class _IntegerLessThan(Constraint):
    """
    Constrain to an integer interval `(-inf, upper_bound]`.
    """
    is_discrete: bool
    upper_bound: Incomplete
    def __init__(self, upper_bound) -> None: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class _IntegerGreaterThan(Constraint):
    """
    Constrain to an integer interval `[lower_bound, inf)`.
    """
    is_discrete: bool
    lower_bound: Incomplete
    def __init__(self, lower_bound) -> None: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class _Real(Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """
    def check(self, value): ...

class _GreaterThan(Constraint):
    """
    Constrain to a real half line `(lower_bound, inf]`.
    """
    lower_bound: Incomplete
    def __init__(self, lower_bound) -> None: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class _GreaterThanEq(Constraint):
    """
    Constrain to a real half line `[lower_bound, inf)`.
    """
    lower_bound: Incomplete
    def __init__(self, lower_bound) -> None: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class _LessThan(Constraint):
    """
    Constrain to a real half line `[-inf, upper_bound)`.
    """
    upper_bound: Incomplete
    def __init__(self, upper_bound) -> None: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class _Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`.
    """
    lower_bound: Incomplete
    upper_bound: Incomplete
    def __init__(self, lower_bound, upper_bound) -> None: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class _HalfOpenInterval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound)`.
    """
    lower_bound: Incomplete
    upper_bound: Incomplete
    def __init__(self, lower_bound, upper_bound) -> None: ...
    def check(self, value): ...
    def __repr__(self) -> str: ...

class _Simplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """
    event_dim: int
    def check(self, value): ...

class _Multinomial(Constraint):
    """
    Constrain to nonnegative integer values summing to at most an upper bound.

    Note due to limitations of the Multinomial distribution, this currently
    checks the weaker condition ``value.sum(-1) <= upper_bound``. In the future
    this may be strengthened to ``value.sum(-1) == upper_bound``.
    """
    is_discrete: bool
    event_dim: int
    upper_bound: Incomplete
    def __init__(self, upper_bound) -> None: ...
    def check(self, x): ...

class _LowerTriangular(Constraint):
    """
    Constrain to lower-triangular square matrices.
    """
    event_dim: int
    def check(self, value): ...

class _LowerCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals.
    """
    event_dim: int
    def check(self, value): ...

class _CorrCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals and each
    row vector being of unit length.
    """
    event_dim: int
    def check(self, value): ...

class _Square(Constraint):
    """
    Constrain to square matrices.
    """
    event_dim: int
    def check(self, value): ...

class _Symmetric(_Square):
    """
    Constrain to Symmetric square matrices.
    """
    def check(self, value): ...

class _PositiveSemidefinite(_Symmetric):
    """
    Constrain to positive-semidefinite matrices.
    """
    def check(self, value): ...

class _PositiveDefinite(_Symmetric):
    """
    Constrain to positive-definite matrices.
    """
    def check(self, value): ...

class _Cat(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
    """
    cseq: Incomplete
    lengths: Incomplete
    dim: Incomplete
    def __init__(self, cseq, dim: int = 0, lengths=None) -> None: ...
    @property
    def is_discrete(self) -> bool: ...
    @property
    def event_dim(self) -> int: ...
    def check(self, value): ...

class _Stack(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    in a way compatible with :func:`torch.stack`.
    """
    cseq: Incomplete
    dim: Incomplete
    def __init__(self, cseq, dim: int = 0) -> None: ...
    @property
    def is_discrete(self) -> bool: ...
    @property
    def event_dim(self) -> int: ...
    def check(self, value): ...

dependent: Incomplete
dependent_property = _DependentProperty
independent = _IndependentConstraint
boolean: Incomplete
one_hot: Incomplete
nonnegative_integer: Incomplete
positive_integer: Incomplete
integer_interval = _IntegerInterval
real: Incomplete
real_vector: Incomplete
positive: Incomplete
nonnegative: Incomplete
greater_than = _GreaterThan
greater_than_eq = _GreaterThanEq
less_than = _LessThan
multinomial = _Multinomial
unit_interval: Incomplete
interval = _Interval
half_open_interval = _HalfOpenInterval
simplex: Incomplete
lower_triangular: Incomplete
lower_cholesky: Incomplete
corr_cholesky: Incomplete
square: Incomplete
symmetric: Incomplete
positive_semidefinite: Incomplete
positive_definite: Incomplete
cat = _Cat
stack = _Stack
