from _typeshed import Incomplete
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import lazy_property
from torch.types import Number, _size

__all__ = ['Wishart']

class Wishart(ExponentialFamily):
    '''
    Creates a Wishart distribution parameterized by a symmetric positive definite matrix :math:`\\Sigma`,
    or its Cholesky decomposition :math:`\\mathbf{\\Sigma} = \\mathbf{L}\\mathbf{L}^\\top`

    Example:
        >>> # xdoctest: +SKIP("FIXME: scale_tril must be at least two-dimensional")
        >>> m = Wishart(torch.Tensor([2]), covariance_matrix=torch.eye(2))
        >>> m.sample()  # Wishart distributed with mean=`df * I` and
        >>> # variance(x_ij)=`df` for i != j and variance(x_ij)=`2 * df` for i == j

    Args:
        df (float or Tensor): real-valued parameter larger than the (dimension of Square matrix) - 1
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.
        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
        \'torch.distributions.LKJCholesky\' is a restricted Wishart distribution.[1]

    **References**

    [1] Wang, Z., Wu, Y. and Chu, H., 2018. `On equivalence of the LKJ distribution and the restricted Wishart distribution`.
    [2] Sawyer, S., 2007. `Wishart Distributions and Inverse-Wishart Sampling`.
    [3] Anderson, T. W., 2003. `An Introduction to Multivariate Statistical Analysis (3rd ed.)`.
    [4] Odell, P. L. & Feiveson, A. H., 1966. `A Numerical Procedure to Generate a SampleCovariance Matrix`. JASA, 61(313):199-203.
    [5] Ku, Y.-C. & Bloomfield, P., 2010. `Generating Random Wishart Matrices with Fractional Degrees of Freedom in OX`.
    '''
    support: Incomplete
    has_rsample: bool
    _mean_carrier_measure: int
    @property
    def arg_constraints(self): ...
    df: Incomplete
    _batch_dims: Incomplete
    _unbroadcasted_scale_tril: Incomplete
    _dist_chi2: Incomplete
    def __init__(self, df: Tensor | Number, covariance_matrix: Tensor | None = None, precision_matrix: Tensor | None = None, scale_tril: Tensor | None = None, validate_args: bool | None = None) -> None: ...
    def expand(self, batch_shape, _instance=None): ...
    @lazy_property
    def scale_tril(self) -> Tensor: ...
    @lazy_property
    def covariance_matrix(self) -> Tensor: ...
    @lazy_property
    def precision_matrix(self) -> Tensor: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def mode(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...
    def _bartlett_sampling(self, sample_shape=...): ...
    def rsample(self, sample_shape: _size = ..., max_try_correction=None) -> Tensor:
        """
        .. warning::
            In some cases, sampling algorithm based on Bartlett decomposition may return singular matrix samples.
            Several tries to correct singular samples are performed by default, but it may end up returning
            singular matrix samples. Singular samples may return `-inf` values in `.log_prob()`.
            In those cases, the user should validate the samples and either fix the value of `df`
            or adjust `max_try_correction` value for argument in `.rsample` accordingly.
        """
    def log_prob(self, value): ...
    def entropy(self): ...
    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]: ...
    def _log_normalizer(self, x, y): ...
