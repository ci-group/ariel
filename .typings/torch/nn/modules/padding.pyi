from .module import Module
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import Tensor
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t

__all__ = ['CircularPad1d', 'CircularPad2d', 'CircularPad3d', 'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d', 'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d', 'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d', 'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d']

class _CircularPadNd(Module):
    __constants__: Incomplete
    padding: Sequence[int]
    def _check_input_dim(self, input) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class CircularPad1d(_CircularPadNd):
    '''Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.CircularPad1d(2)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[2., 3., 0., 1., 2., 3., 0., 1.],
                 [6., 7., 4., 5., 6., 7., 4., 5.]]])
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad1d((3, 1))
        >>> m(input)
        tensor([[[1., 2., 3., 0., 1., 2., 3., 0.],
                 [5., 6., 7., 4., 5., 6., 7., 4.]]])
    '''
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...
    def _check_input_dim(self, input) -> None: ...

class CircularPad2d(_CircularPadNd):
    """Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\\text{padding\\_left}`,
            :math:`\\text{padding\\_right}`, :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> m = nn.CircularPad2d(2)
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[4., 5., 3., 4., 5., 3., 4.],
                  [7., 8., 6., 7., 8., 6., 7.],
                  [1., 2., 0., 1., 2., 0., 1.],
                  [4., 5., 3., 4., 5., 3., 4.],
                  [7., 8., 6., 7., 8., 6., 7.],
                  [1., 2., 0., 1., 2., 0., 1.],
                  [4., 5., 3., 4., 5., 3., 4.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[5., 3., 4., 5., 3.],
                  [8., 6., 7., 8., 6.],
                  [2., 0., 1., 2., 0.],
                  [5., 3., 4., 5., 3.],
                  [8., 6., 7., 8., 6.]]]])
    """
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...
    def _check_input_dim(self, input) -> None: ...

class CircularPad3d(_CircularPadNd):
    '''Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`,
            :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`,
            :math:`\\text{padding\\_front}`, :math:`\\text{padding\\_back}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \\text{padding\\_front} + \\text{padding\\_back}`

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.CircularPad3d(3)
        >>> input = torch.randn(16, 3, 8, 320, 480)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)
    '''
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...
    def _check_input_dim(self, input) -> None: ...

class _ConstantPadNd(Module):
    __constants__: Incomplete
    value: float
    padding: Sequence[int]
    def __init__(self, value: float) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ConstantPad1d(_ConstantPadNd):
    '''Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000, -1.0491, -0.7152, -0.0749,  0.8530,  3.5000,
                   3.5000],
                 [ 3.5000,  3.5000, -1.3287,  1.8966,  0.1466, -0.2771,  3.5000,
                   3.5000]]])
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad1d((3, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000],
                 [ 3.5000,  3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000]]])
    '''
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t, value: float) -> None: ...

class ConstantPad2d(_ConstantPadNd):
    '''Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\\text{padding\\_left}`,
            :math:`\\text{padding\\_right}`, :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ConstantPad2d(2, 3.5)
        >>> input = torch.randn(1, 2, 2)
        >>> input
        tensor([[[ 1.6585,  0.4320],
                 [-0.8701, -0.4649]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  1.6585,  0.4320,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -0.8701, -0.4649,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  1.6585,  0.4320],
                 [ 3.5000,  3.5000,  3.5000, -0.8701, -0.4649],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
    '''
    __constants__: Incomplete
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t, value: float) -> None: ...

class ConstantPad3d(_ConstantPadNd):
    """Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`,
            :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`,
            :math:`\\text{padding\\_front}`, :math:`\\text{padding\\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          :math:`D_{out} = D_{in} + \\text{padding\\_front} + \\text{padding\\_back}`

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> m = nn.ConstantPad3d(3, 3.5)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
        >>> output = m(input)
    """
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t, value: float) -> None: ...

class _ReflectionPadNd(Module):
    __constants__: Incomplete
    padding: Sequence[int]
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ReflectionPad1d(_ReflectionPadNd):
    '''Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> m = nn.ReflectionPad1d(2)
        >>> # xdoctest: +IGNORE_WANT("other tests seem to modify printing styles")
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[2., 1., 0., 1., 2., 3., 2., 1.],
                 [6., 5., 4., 5., 6., 7., 6., 5.]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReflectionPad1d((3, 1))
        >>> m(input)
        tensor([[[3., 2., 1., 0., 1., 2., 3., 2.],
                 [7., 6., 5., 4., 5., 6., 7., 6.]]])
    '''
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...

class ReflectionPad2d(_ReflectionPadNd):
    '''Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\\text{padding\\_left}`,
            :math:`\\text{padding\\_right}`, :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})` where

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReflectionPad2d(2)
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[8., 7., 6., 7., 8., 7., 6.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [2., 1., 0., 1., 2., 1., 0.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [8., 7., 6., 7., 8., 7., 6.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [2., 1., 0., 1., 2., 1., 0.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReflectionPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[7., 6., 7., 8., 7.],
                  [4., 3., 4., 5., 4.],
                  [1., 0., 1., 2., 1.],
                  [4., 3., 4., 5., 4.],
                  [7., 6., 7., 8., 7.]]]])
    '''
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...

class ReflectionPad3d(_ReflectionPadNd):
    '''Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`,
            :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`,
            :math:`\\text{padding\\_front}`, :math:`\\text{padding\\_back}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \\text{padding\\_front} + \\text{padding\\_back}`

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReflectionPad3d(1)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2)
        >>> m(input)
        tensor([[[[[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]],
                  [[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]]]]])
    '''
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...

class _ReplicationPadNd(Module):
    __constants__: Incomplete
    padding: Sequence[int]
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ReplicationPad1d(_ReplicationPadNd):
    '''Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReplicationPad1d(2)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[0., 0., 0., 1., 2., 3., 3., 3.],
                 [4., 4., 4., 5., 6., 7., 7., 7.]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad1d((3, 1))
        >>> m(input)
        tensor([[[0., 0., 0., 0., 1., 2., 3., 3.],
                 [4., 4., 4., 4., 5., 6., 7., 7.]]])
    '''
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...

class ReplicationPad2d(_ReplicationPadNd):
    '''Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\\text{padding\\_left}`,
            :math:`\\text{padding\\_right}`, :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> m = nn.ReplicationPad2d(2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[0., 0., 0., 1., 2., 2., 2.],
                  [0., 0., 0., 1., 2., 2., 2.],
                  [0., 0., 0., 1., 2., 2., 2.],
                  [3., 3., 3., 4., 5., 5., 5.],
                  [6., 6., 6., 7., 8., 8., 8.],
                  [6., 6., 6., 7., 8., 8., 8.],
                  [6., 6., 6., 7., 8., 8., 8.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[0., 0., 1., 2., 2.],
                  [0., 0., 1., 2., 2.],
                  [0., 0., 1., 2., 2.],
                  [3., 3., 4., 5., 5.],
                  [6., 6., 7., 8., 8.]]]])
    '''
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...

class ReplicationPad3d(_ReplicationPadNd):
    '''Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`,
            :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`,
            :math:`\\text{padding\\_front}`, :math:`\\text{padding\\_back}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \\text{padding\\_front} + \\text{padding\\_back}`

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ReplicationPad3d(3)
        >>> input = torch.randn(16, 3, 8, 320, 480)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)
    '''
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...

class ZeroPad1d(ConstantPad1d):
    '''Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ZeroPad1d(2)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 0.0000,  0.0000, -1.0491, -0.7152, -0.0749,  0.8530,  0.0000,
                   0.0000],
                 [ 0.0000,  0.0000, -1.3287,  1.8966,  0.1466, -0.2771,  0.0000,
                   0.0000]]])
        >>> m = nn.ZeroPad1d(2)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000,  0.0000],
                 [ 0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000,  0.0000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad1d((3, 1))
        >>> m(input)
        tensor([[[ 0.0000,  0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000],
                 [ 0.0000,  0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000]]])
    '''
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...
    def extra_repr(self) -> str: ...

class ZeroPad2d(ConstantPad2d):
    '''Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\\text{padding\\_left}`,
            :math:`\\text{padding\\_right}`, :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ZeroPad2d(2)
        >>> input = torch.randn(1, 1, 3, 3)
        >>> input
        tensor([[[[-0.1678, -0.4418,  1.9466],
                  [ 0.9604, -0.4219, -0.5241],
                  [-0.9162, -0.5436, -0.6446]]]])
        >>> m(input)
        tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000, -0.1678, -0.4418,  1.9466,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.9604, -0.4219, -0.5241,  0.0000,  0.0000],
                  [ 0.0000,  0.0000, -0.9162, -0.5436, -0.6446,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000, -0.1678, -0.4418,  1.9466,  0.0000],
                  [ 0.0000,  0.9604, -0.4219, -0.5241,  0.0000],
                  [ 0.0000, -0.9162, -0.5436, -0.6446,  0.0000]]]])
    '''
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...
    def extra_repr(self) -> str: ...

class ZeroPad3d(ConstantPad3d):
    """Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\\text{padding\\_left}`, :math:`\\text{padding\\_right}`,
            :math:`\\text{padding\\_top}`, :math:`\\text{padding\\_bottom}`,
            :math:`\\text{padding\\_front}`, :math:`\\text{padding\\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          :math:`D_{out} = D_{in} + \\text{padding\\_front} + \\text{padding\\_back}`

          :math:`H_{out} = H_{in} + \\text{padding\\_top} + \\text{padding\\_bottom}`

          :math:`W_{out} = W_{in} + \\text{padding\\_left} + \\text{padding\\_right}`

    Examples::

        >>> m = nn.ZeroPad3d(3)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))
        >>> output = m(input)
    """
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...
    def extra_repr(self) -> str: ...
