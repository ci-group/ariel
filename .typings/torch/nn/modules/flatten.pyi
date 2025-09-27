from .module import Module
from _typeshed import Incomplete
from torch import Tensor
from torch.types import _size

__all__ = ['Flatten', 'Unflatten']

class Flatten(Module):
    """
    Flattens a contiguous range of dims into a tensor.

    For use with :class:`~nn.Sequential`, see :meth:`torch.flatten` for details.

    Shape:
        - Input: :math:`(*, S_{\\text{start}},..., S_{i}, ..., S_{\\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \\prod_{i=\\text{start}}^{\\text{end}} S_{i}, *)`.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Examples::
        >>> input = torch.randn(32, 1, 5, 5)
        >>> # With default parameters
        >>> m = nn.Flatten()
        >>> output = m(input)
        >>> output.size()
        torch.Size([32, 25])
        >>> # With non-default parameters
        >>> m = nn.Flatten(0, 2)
        >>> output = m(input)
        >>> output.size()
        torch.Size([160, 5])
    """
    __constants__: Incomplete
    start_dim: int
    end_dim: int
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Unflatten(Module):
    '''
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it can
      be either `int` or `str` when `Tensor` or `NamedTensor` is used, respectively.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `torch.Size` for `Tensor` input;  a `NamedShape`
      (tuple of `(name, size)` tuples) for `NamedTensor` input.

    Shape:
        - Input: :math:`(*, S_{\\text{dim}}, *)`, where :math:`S_{\\text{dim}}` is the size at
          dimension :attr:`dim` and :math:`*` means any number of dimensions including none.
        - Output: :math:`(*, U_1, ..., U_n, *)`, where :math:`U` = :attr:`unflattened_size` and
          :math:`\\prod_{i=1}^n U_i = S_{\\text{dim}}`.

    Args:
        dim (Union[int, str]): Dimension to be unflattened
        unflattened_size (Union[torch.Size, Tuple, List, NamedShape]): New shape of the unflattened dimension

    Examples:
        >>> input = torch.randn(2, 50)
        >>> # With tuple of ints
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With torch.Size
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, torch.Size([2, 5, 5]))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With namedshape (tuple of tuples)
        >>> input = torch.randn(2, 50, names=("N", "features"))
        >>> unflatten = nn.Unflatten("features", (("C", 2), ("H", 5), ("W", 5)))
        >>> output = unflatten(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
    '''
    NamedShape = tuple[tuple[str, int]]
    __constants__: Incomplete
    dim: int | str
    unflattened_size: _size | NamedShape
    def __init__(self, dim: int | str, unflattened_size: _size | NamedShape) -> None: ...
    def _require_tuple_tuple(self, input) -> None: ...
    def _require_tuple_int(self, input) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
