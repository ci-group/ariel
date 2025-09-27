from _typeshed import Incomplete
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions.exp_family import ExponentialFamily
from torch.types import _size

__all__ = ['Dirichlet']

class _Dirichlet(Function):
    @staticmethod
    def forward(ctx, concentration): ...
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output): ...

class Dirichlet(ExponentialFamily):
    '''
    Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]
        tensor([ 0.1046,  0.8954])

    Args:
        concentration (Tensor): concentration parameter of the distribution
            (often referred to as alpha)
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    concentration: Incomplete
    def __init__(self, concentration: Tensor, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def rsample(self, sample_shape: _size = ()) -> Tensor: ...
    def log_prob(self, value): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def entropy(self): ...
    @property
    def _natural_params(self) -> tuple[Tensor]: ...
    def _log_normalizer(self, x): ...
