from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property
from torch.types import _size

__all__ = ['LowRankMultivariateNormal']

class LowRankMultivariateNormal(Distribution):
    '''
    Creates a multivariate normal distribution with covariance matrix having a low-rank form
    parameterized by :attr:`cov_factor` and :attr:`cov_diag`::

        covariance_matrix = cov_factor @ cov_factor.T + cov_diag

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LowRankMultivariateNormal(
        ...     torch.zeros(2), torch.tensor([[1.0], [0.0]]), torch.ones(2)
        ... )
        >>> m.sample()  # normally distributed with mean=`[0,0]`, cov_factor=`[[1],[0]]`, cov_diag=`[1,1]`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution with shape `batch_shape + event_shape`
        cov_factor (Tensor): factor part of low-rank form of covariance matrix with shape
            `batch_shape + event_shape + (rank,)`
        cov_diag (Tensor): diagonal part of low-rank form of covariance matrix with shape
            `batch_shape + event_shape`

    Note:
        The computation for determinant and inverse of covariance matrix is avoided when
        `cov_factor.shape[1] << cov_factor.shape[0]` thanks to `Woodbury matrix identity
        <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_ and
        `matrix determinant lemma <https://en.wikipedia.org/wiki/Matrix_determinant_lemma>`_.
        Thanks to these formulas, we just need to compute the determinant and inverse of
        the small size "capacitance" matrix::

            capacitance = I + cov_factor.T @ inv(cov_diag) @ cov_factor
    '''
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    loc: Incomplete
    cov_diag: Incomplete
    _unbroadcasted_cov_factor: Incomplete
    _unbroadcasted_cov_diag: Incomplete
    _capacitance_tril: Incomplete
    def __init__(self, loc: Tensor, cov_factor: Tensor, cov_diag: Tensor, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @lazy_property
    def variance(self) -> Tensor: ...
    @lazy_property
    def scale_tril(self) -> Tensor: ...
    @lazy_property
    def covariance_matrix(self) -> Tensor: ...
    @lazy_property
    def precision_matrix(self) -> Tensor: ...
    def rsample(self, sample_shape: _size = ...) -> Tensor: ...
    def log_prob(self, value): ...
    def entropy(self): ...
