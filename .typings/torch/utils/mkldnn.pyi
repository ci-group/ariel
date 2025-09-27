import torch
from _typeshed import Incomplete

class MkldnnLinear(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype) -> None: ...
    @torch.jit.script_method
    def __getstate__(self): ...
    weight: Incomplete
    bias: Incomplete
    training: Incomplete
    @torch.jit.script_method
    def __setstate__(self, state) -> None: ...
    @torch.jit.script_method
    def forward(self, x): ...

class _MkldnnConvNd(torch.jit.ScriptModule):
    """Common base of MkldnnConv1d and MkldnnConv2d."""
    __constants__: Incomplete
    stride: Incomplete
    padding: Incomplete
    dilation: Incomplete
    groups: Incomplete
    def __init__(self, dense_module) -> None: ...
    @torch.jit.script_method
    def __getstate__(self): ...
    @torch.jit.script_method
    def forward(self, x): ...

class MkldnnConv1d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None: ...
    weight: Incomplete
    bias: Incomplete
    training: Incomplete
    @torch.jit.script_method
    def __setstate__(self, state) -> None: ...

class MkldnnConv2d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None: ...
    weight: Incomplete
    bias: Incomplete
    training: Incomplete
    @torch.jit.script_method
    def __setstate__(self, state) -> None: ...

class MkldnnConv3d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None: ...
    weight: Incomplete
    bias: Incomplete
    training: Incomplete
    @torch.jit.script_method
    def __setstate__(self, state) -> None: ...

class MkldnnBatchNorm(torch.jit.ScriptModule):
    __constants__: Incomplete
    exponential_average_factor: float
    eps: Incomplete
    def __init__(self, dense_module) -> None: ...
    @torch.jit.script_method
    def __getstate__(self): ...
    weight: Incomplete
    bias: Incomplete
    running_mean: Incomplete
    running_var: Incomplete
    training: Incomplete
    @torch.jit.script_method
    def __setstate__(self, state) -> None: ...
    @torch.jit.script_method
    def forward(self, x): ...

class MkldnnPrelu(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype) -> None: ...
    @torch.jit.script_method
    def __getstate__(self): ...
    weight: Incomplete
    training: Incomplete
    @torch.jit.script_method
    def __setstate__(self, state) -> None: ...
    @torch.jit.script_method
    def forward(self, x): ...

def to_mkldnn(module, dtype=...): ...
