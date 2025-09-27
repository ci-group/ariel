from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution

__all__ = ['LKJCholesky']

class LKJCholesky(Distribution):
    '''
    LKJ distribution for lower Cholesky factor of correlation matrices.
    The distribution is controlled by ``concentration`` parameter :math:`\\eta`
    to make the probability of the correlation matrix :math:`M` generated from
    a Cholesky factor proportional to :math:`\\det(M)^{\\eta - 1}`. Because of that,
    when ``concentration == 1``, we have a uniform distribution over Cholesky
    factors of correlation matrices::

        L ~ LKJCholesky(dim, concentration)
        X = L @ L\' ~ LKJCorr(dim, concentration)

    Note that this distribution samples the
    Cholesky factor of correlation matrices and not the correlation matrices
    themselves and thereby differs slightly from the derivations in [1] for
    the `LKJCorr` distribution. For sampling, this uses the Onion method from
    [1] Section 3.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> l = LKJCholesky(3, 0.5)
        >>> l.sample()  # l @ l.T is a sample of a correlation 3x3 matrix
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.3516,  0.9361,  0.0000],
                [-0.1899,  0.4748,  0.8593]])

    Args:
        dimension (dim): dimension of the matrices
        concentration (float or Tensor): concentration/shape parameter of the
            distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method` (2009),
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    Journal of Multivariate Analysis. 100. 10.1016/j.jmva.2009.04.008
    '''
    arg_constraints: Incomplete
    support: Incomplete
    dim: Incomplete
    _beta: Incomplete
    def __init__(self, dim: int, concentration: Tensor | float = 1.0, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
