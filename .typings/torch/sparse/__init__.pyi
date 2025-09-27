import types
from .semi_structured import SparseSemiStructuredTensor as SparseSemiStructuredTensor, SparseSemiStructuredTensorCUSPARSELT as SparseSemiStructuredTensorCUSPARSELT, SparseSemiStructuredTensorCUTLASS as SparseSemiStructuredTensorCUTLASS, to_sparse_semi_structured as to_sparse_semi_structured
from _typeshed import Incomplete
from torch import Tensor
from torch.types import _dtype as DType

__all__ = ['addmm', 'check_sparse_tensor_invariants', 'mm', 'sum', 'softmax', 'solve', 'log_softmax', 'SparseSemiStructuredTensor', 'SparseSemiStructuredTensorCUTLASS', 'SparseSemiStructuredTensorCUSPARSELT', 'to_sparse_semi_structured', 'as_sparse_gradcheck']

DimOrDims = int | tuple[int, ...] | list[int] | None
addmm: Incomplete
mm: Incomplete

def sum(input: Tensor, dim: DimOrDims = None, dtype: DType | None = None) -> Tensor:
    '''Return the sum of each row of the given sparse tensor.

    Returns the sum of each row of the sparse tensor :attr:`input` in the given
    dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them. When sum over all ``sparse_dim``, this method
    returns a dense tensor instead of a sparse tensor.

    All summed :attr:`dim` are squeezed (see :func:`torch.squeeze`), resulting an output
    tensor having :attr:`dim` fewer dimensions than :attr:`input`.

    During backward, only gradients at ``nnz`` locations of :attr:`input`
    will propagate back. Note that the gradients of :attr:`input` is coalesced.

    Args:
        input (Tensor): the input sparse tensor
        dim (int or tuple of ints): a dimension or a list of dimensions to reduce. Default: reduce
            over all dims.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
            Default: dtype of :attr:`input`.

    Example::

        >>> nnz = 3
        >>> dims = [5, 5, 2, 3]
        >>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                           torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
        >>> V = torch.randn(nnz, dims[2], dims[3])
        >>> size = torch.Size(dims)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> S = torch.sparse_coo_tensor(I, V, size)
        >>> S
        tensor(indices=tensor([[2, 0, 3],
                               [2, 4, 1]]),
               values=tensor([[[-0.6438, -1.6467,  1.4004],
                               [ 0.3411,  0.0918, -0.2312]],

                              [[ 0.5348,  0.0634, -2.0494],
                               [-0.7125, -1.0646,  2.1844]],

                              [[ 0.1276,  0.1874, -0.6334],
                               [-1.9682, -0.5340,  0.7483]]]),
               size=(5, 5, 2, 3), nnz=3, layout=torch.sparse_coo)

        # when sum over only part of sparse_dims, return a sparse tensor
        >>> torch.sparse.sum(S, [1, 3])
        tensor(indices=tensor([[0, 2, 3]]),
               values=tensor([[-1.4512,  0.4073],
                              [-0.8901,  0.2017],
                              [-0.3183, -1.7539]]),
               size=(5, 2), nnz=3, layout=torch.sparse_coo)

        # when sum over all sparse dim, return a dense tensor
        # with summed dims squeezed
        >>> torch.sparse.sum(S, [0, 1, 3])
        tensor([-2.6596, -1.1450])
    '''

softmax: Incomplete
log_softmax: Incomplete

class check_sparse_tensor_invariants:
    '''A tool to control checking sparse tensor invariants.

    The following options exists to manage sparsr tensor invariants
    checking in sparse tensor construction:

    1. Using a context manager:

       .. code:: python

           with torch.sparse.check_sparse_tensor_invariants():
               run_my_model()

    2. Using a procedural approach:

       .. code:: python

           prev_checks_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
           torch.sparse.check_sparse_tensor_invariants.enable()

           run_my_model()

           if not prev_checks_enabled:
               torch.sparse.check_sparse_tensor_invariants.disable()

    3. Using function decoration:

       .. code:: python

           @torch.sparse.check_sparse_tensor_invariants()
           def run_my_model():
               ...

           run_my_model()

    4. Using ``check_invariants`` keyword argument in sparse tensor constructor call.
       For example:

       >>> torch.sparse_csr_tensor([0, 1, 3], [0, 1], [1, 2], check_invariants=True)
       Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
       RuntimeError: `crow_indices[..., -1] == nnz` is not satisfied.
    '''
    @staticmethod
    def is_enabled():
        """Return True if the sparse tensor invariants checking is enabled.

        .. note::

            Use :func:`torch.sparse.check_sparse_tensor_invariants.enable` or
            :func:`torch.sparse.check_sparse_tensor_invariants.disable` to
            manage the state of the sparse tensor invariants checks.
        """
    @staticmethod
    def enable() -> None:
        """Enable sparse tensor invariants checking in sparse tensor constructors.

        .. note::

            By default, the sparse tensor invariants checks are disabled. Use
            :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled` to
            retrieve the current state of sparse tensor invariants checking.

        .. note::

            The sparse tensor invariants check flag is effective to all sparse
            tensor constructors, both in Python and ATen.

        The flag can be locally overridden by the ``check_invariants``
        optional argument of the sparse tensor constructor functions.
        """
    @staticmethod
    def disable() -> None:
        """Disable sparse tensor invariants checking in sparse tensor constructors.

        See :func:`torch.sparse.check_sparse_tensor_invariants.enable` for more information.
        """
    state: Incomplete
    saved_state: bool | None
    def __init__(self, enable: bool = True) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def __call__(self, mth): ...

def as_sparse_gradcheck(gradcheck):
    """Decorate function, to extend gradcheck for sparse tensors.

    Decorator for torch.autograd.gradcheck or its functools.partial
    variants that extends the gradcheck function with support to input
    functions that operate on or/and return sparse tensors.

    The specified gradcheck function itself is guaranteed to operate
    on strided tensors only.

    For example:

    >>> gradcheck = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)
    >>> x = torch.tensor([[0, 1], [2, 3]], dtype=torch.float64).to_sparse_coo().requires_grad_(True)
    >>> gradcheck(lambda x: x.to_sparse_csr(), x)
    True
    """

# Names in __all__ with no definition:
#   solve
