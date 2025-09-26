from .distribution import Distribution
from _typeshed import Incomplete
from torch import Tensor

__all__ = ['register_kl', 'kl_divergence']

def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """

class _Match:
    __slots__: Incomplete
    types: Incomplete
    def __init__(self, *types) -> None: ...
    def __eq__(self, other): ...
    def __le__(self, other): ...

def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    """
    Compute Kullback-Leibler divergence :math:`KL(p \\| q)` between two distributions.

    .. math::

        KL(p \\| q) = \\int p(x) \\log\\frac {p(x)} {q(x)} \\,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
