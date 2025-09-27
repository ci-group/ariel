import torch
from _typeshed import Incomplete

__all__ = ['ConvReLU1d', 'ConvReLU2d', 'ConvReLU3d', 'LinearReLU', 'ConvBn1d', 'ConvBn2d', 'ConvBnReLU1d', 'ConvBnReLU2d', 'ConvBn3d', 'ConvBnReLU3d', 'BNReLU2d', 'BNReLU3d', 'LinearBn1d', 'LinearLeakyReLU', 'LinearTanh', 'ConvAdd2d', 'ConvAddReLU2d']

class _FusedModule(torch.nn.Sequential): ...

class ConvReLU1d(_FusedModule):
    """This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu) -> None: ...

class ConvReLU2d(_FusedModule):
    """This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu) -> None: ...

class ConvReLU3d(_FusedModule):
    """This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu) -> None: ...

class LinearReLU(_FusedModule):
    """This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, relu) -> None: ...

class ConvBn1d(_FusedModule):
    """This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn) -> None: ...

class ConvBn2d(_FusedModule):
    """This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn) -> None: ...

class ConvBnReLU1d(_FusedModule):
    """This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu) -> None: ...

class ConvBnReLU2d(_FusedModule):
    """This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu) -> None: ...

class ConvBn3d(_FusedModule):
    """This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn) -> None: ...

class ConvBnReLU3d(_FusedModule):
    """This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu) -> None: ...

class BNReLU2d(_FusedModule):
    """This is a sequential container which calls the BatchNorm 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, batch_norm, relu) -> None: ...

class BNReLU3d(_FusedModule):
    """This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, batch_norm, relu) -> None: ...

class LinearBn1d(_FusedModule):
    """This is a sequential container which calls the Linear and BatchNorm1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, bn) -> None: ...

class LinearLeakyReLU(_FusedModule):
    """This is a sequential container which calls the Linear and LeakyReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, leaky_relu) -> None: ...

class LinearTanh(_FusedModule):
    """This is a sequential container which calls the Linear and Tanh modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, tanh) -> None: ...

class ConvAdd2d(_FusedModule):
    """This is a sequential container which calls the Conv2d modules with extra Add.
    During quantization this will be replaced with the corresponding fused module."""
    add: Incomplete
    def __init__(self, conv, add) -> None: ...
    def forward(self, x1, x2): ...

class ConvAddReLU2d(_FusedModule):
    """This is a sequential container which calls the Conv2d, add, Relu.
    During quantization this will be replaced with the corresponding fused module."""
    add: Incomplete
    relu: Incomplete
    def __init__(self, conv, add, relu) -> None: ...
    def forward(self, x1, x2): ...
