import functools
import torch
from ..pattern_matcher import fwd_only as fwd_only, register_replacement as register_replacement
from _typeshed import Incomplete
from torch._dynamo.utils import counters as counters
from torch._ops import OpOverload as OpOverload, OpOverloadPacket as OpOverloadPacket
from torch.utils._ordered_set import OrderedSet as OrderedSet

aten: Incomplete

@functools.cache
def _misc_patterns_init(): ...

class NumpyCompatNormalization:
    numpy_compat: dict[str, tuple[str, ...]]
    inverse_mapping: dict[str, str]
    cache: dict['torch.fx.graph.Target', OrderedSet[str]]
    def __init__(self) -> None: ...
    def __call__(self, graph: torch.fx.Graph): ...

numpy_compat_normalization: Incomplete
