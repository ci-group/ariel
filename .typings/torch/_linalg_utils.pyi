from torch import Tensor as Tensor

def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
def get_floating_dtype(A):
    """Return the floating point dtype of tensor A.

    Integer types map to float32.
    """
def matmul(A: Tensor | None, B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
def bform(X: Tensor, A: Tensor | None, Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
def qform(A: Tensor | None, S: Tensor):
    """Return quadratic form :math:`S^T A S`."""
def basis(A):
    """Return orthogonal basis of A columns."""
def symeig(A: Tensor, largest: bool | None = False) -> tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""
def matrix_rank(input, tol=None, symmetric: bool = False, *, out=None) -> Tensor: ...
def solve(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]: ...
def lstsq(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]: ...
def _symeig(input, eigenvectors: bool = False, upper: bool = True, *, out=None) -> tuple[Tensor, Tensor]: ...
def eig(self, eigenvectors: bool = False, *, e=None, v=None) -> tuple[Tensor, Tensor]: ...
