from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.gamma import Gamma

__all__ = ['Chi2']

class Chi2(Gamma):
    '''
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi2(torch.tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    '''
    arg_constraints: Incomplete
    def __init__(self, df: Tensor | float, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def df(self) -> Tensor: ...
