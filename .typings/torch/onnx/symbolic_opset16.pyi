import torch
from _typeshed import Incomplete
from torch.nn.functional import GRID_SAMPLE_INTERPOLATION_MODES as GRID_SAMPLE_INTERPOLATION_MODES, GRID_SAMPLE_PADDING_MODES as GRID_SAMPLE_PADDING_MODES
from torch.onnx import _type_utils as _type_utils, errors as errors, symbolic_helper as symbolic_helper, utils as utils
from torch.onnx._internal import jit_utils as jit_utils, registration as registration

_onnx_symbolic: Incomplete

def grid_sampler(g: jit_utils.GraphContext, input, grid, mode_enum, padding_mode_enum, align_corners): ...
def scatter_add(g: jit_utils.GraphContext, self, dim, index, src): ...
def scatter_reduce(g: jit_utils.GraphContext, self: torch._C.Value, dim: int, index: torch._C.Value, src: torch._C.Value, reduce: str, include_self: bool): ...
