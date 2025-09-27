from collections.abc import Sequence
from torch import _C

__all__ = ['col2im']

def col2im(g, input: _C.Value, output_size: _C.Value, kernel_size: _C.Value, dilation: Sequence[int], padding: Sequence[int], stride: Sequence[int]): ...
