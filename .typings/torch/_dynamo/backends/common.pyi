import torch
from _typeshed import Incomplete
from torch._dynamo import disable as disable
from torch._dynamo.exc import TensorifyScalarRestartAnalysis as TensorifyScalarRestartAnalysis
from torch._dynamo.utils import counters as counters, defake as defake, flatten_graph_inputs as flatten_graph_inputs
from torch._functorch.aot_autograd import SerializableAOTDispatchCompiler as SerializableAOTDispatchCompiler, aot_module_simplified as aot_module_simplified
from torch.utils._python_dispatch import _disable_current_modes as _disable_current_modes

log: Incomplete

class AotAutograd:
    __name__: str
    kwargs: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def __call__(self, gm: torch.fx.GraphModule, example_inputs, **kwargs): ...

def aot_autograd(**kwargs) -> AotAutograd: ...
def mem_efficient_fusion_kwargs(use_decomps): ...
def fake_tensor_unsupported(fn):
    """
    Decorator for backends that need real inputs.  We swap out fake
    tensors for zero tensors.
    """
def device_from_inputs(example_inputs) -> torch.device: ...
def dtype_from_inputs(example_inputs) -> torch.dtype: ...
