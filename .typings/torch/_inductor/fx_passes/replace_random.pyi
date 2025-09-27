import torch
from .. import config as config, inductor_prims as inductor_prims
from ..pattern_matcher import CallFunctionVarArgs as CallFunctionVarArgs, Match as Match, PatternMatcherPass as PatternMatcherPass, register_graph_pattern as register_graph_pattern
from ..virtualized import V as V
from _typeshed import Incomplete
from torch.fx.passes.graph_transform_observer import GraphTransformObserver as GraphTransformObserver
from torch.fx.passes.shape_prop import _extract_tensor_metadata as _extract_tensor_metadata

log: Incomplete
patterns: Incomplete
aten: Incomplete

def replace_random_passes(gm: torch.fx.GraphModule):
    """Modify the given FX graph to use backend-native random ops"""
def fuse_seed_creation_pass(graph: torch.fx.Graph):
    """
    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
def default_kwargs(device): ...
def get_device(device): ...
def replace_random(match: Match, size, *, generator=None, dtype=None, device=None, layout=None, pin_memory=None): ...
def replace_randint(match: Match, low, high, size, *, dtype=..., device=None, layout=None, pin_memory=None): ...
