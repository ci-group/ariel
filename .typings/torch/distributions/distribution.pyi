import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributions import constraints
from torch.types import _size

__all__ = ['Distribution']

class Distribution:
    """
    Distribution is the abstract base class for probability distributions.

    Args:
        batch_shape (torch.Size): The shape over which parameters are batched.
        event_shape (torch.Size): The shape of a single sample (without batching).
        validate_args (bool, optional): Whether to validate arguments. Default: None.
    """
    has_rsample: bool
    has_enumerate_support: bool
    _validate_args = __debug__
    @staticmethod
    def set_default_validate_args(value: bool) -> None:
        """
        Sets whether validation is enabled or disabled.

        The default behavior mimics Python's ``assert`` statement: validation
        is on by default, but is disabled if Python is run in optimized mode
        (via ``python -O``). Validation may be expensive, so you may want to
        disable it once a model is working.

        Args:
            value (bool): Whether to enable validation.
        """
    _batch_shape: Incomplete
    _event_shape: Incomplete
    def __init__(self, batch_shape: torch.Size = ..., event_shape: torch.Size = ..., validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape: _size, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the shape over which parameters are batched.
        """
    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the shape of a single sample (without batching).
        """
    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """
        Returns a dictionary from argument names to
        :class:`~torch.distributions.constraints.Constraint` objects that
        should be satisfied by each argument of this distribution. Args that
        are not tensors need not appear in this dict.
        """
    @property
    def support(self) -> constraints.Constraint | None:
        """
        Returns a :class:`~torch.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
    @property
    def mean(self) -> Tensor:
        """
        Returns the mean of the distribution.
        """
    @property
    def mode(self) -> Tensor:
        """
        Returns the mode of the distribution.
        """
    @property
    def variance(self) -> Tensor:
        """
        Returns the variance of the distribution.
        """
    @property
    def stddev(self) -> Tensor:
        """
        Returns the standard deviation of the distribution.
        """
    def sample(self, sample_shape: _size = ...) -> Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
    def rsample(self, sample_shape: _size = ...) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
    def sample_n(self, n: int) -> Tensor:
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
    def log_prob(self, value: Tensor) -> Tensor:
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
    def cdf(self, value: Tensor) -> Tensor:
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
    def icdf(self, value: Tensor) -> Tensor:
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
    def enumerate_support(self, expand: bool = True) -> Tensor:
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        """
    def entropy(self) -> Tensor:
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
    def perplexity(self) -> Tensor:
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
    def _extended_shape(self, sample_shape: _size = ...) -> torch.Size:
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (torch.Size): the size of the sample to be drawn.
        """
    def _validate_sample(self, value: Tensor) -> None:
        """
        Argument validation for distribution methods such as `log_prob`,
        `cdf` and `icdf`. The rightmost dimensions of a value to be
        scored via these methods must agree with the distribution's batch
        and event shapes.

        Args:
            value (Tensor): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
    def _get_checked_instance(self, cls, _instance=None): ...
    def __repr__(self) -> str: ...
