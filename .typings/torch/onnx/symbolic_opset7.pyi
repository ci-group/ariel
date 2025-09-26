from _typeshed import Incomplete
from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx._internal import jit_utils as jit_utils, registration as registration

_onnx_symbolic: Incomplete
block_listed_operators: Incomplete

def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None): ...
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None): ...
