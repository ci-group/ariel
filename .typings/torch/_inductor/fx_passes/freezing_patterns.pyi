import functools
import torch
from .. import config as config
from ..._dynamo.utils import counters as counters
from ..pattern_matcher import CallFunction as CallFunction, Ignored as Ignored, KeywordArg as KeywordArg, Match as Match, PatternMatcherPass as PatternMatcherPass, _return_true as _return_true, fwd_only as fwd_only, init_once_fakemode as init_once_fakemode, register_graph_pattern as register_graph_pattern, register_replacement as register_replacement, stable_topological_sort as stable_topological_sort
from _typeshed import Incomplete
from torch._inductor.compile_fx import fake_tensor_prop as fake_tensor_prop
from torch._inductor.utils import GPU_TYPES as GPU_TYPES

aten: Incomplete
pass_patterns: Incomplete
binary_folding_pass: Incomplete

def freezing_passes(gm: torch.fx.GraphModule, aot_example_inputs):
    """
    Passes that are applied to the graph to freeze pass.
    """
@init_once_fakemode
def lazy_init() -> None: ...
def register_freezing_graph_pattern(pattern, extra_check=..., pass_number: int = 0): ...
def register_binary_folding_pattern(pattern, extra_check=...): ...
@functools.cache
def addmm_patterns_init():
    """
    addmm related patterns.
    To avoid duplication, also includes int8 WoQ GEMM pattern without bias.
    """
def same_dtype(match): ...
def unnecessary_dtype_convert(match: Match, **kwargs):
    """Remove unnecessary dtype conversion op, probably left as a result of Conv-Bn folding"""
