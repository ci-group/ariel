import torch
import torch.fx as fx
from .compile_utils import get_outputs as get_outputs, get_placeholders as get_placeholders
from _typeshed import Incomplete
from dataclasses import dataclass
from torch.hub import tqdm as tqdm
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.utils._content_store import ContentStoreWriter as ContentStoreWriter
from typing import Callable

is_tuple: Incomplete

@dataclass
class LoadTensorMeta:
    size: list[int]
    stride: list[int]
    dtype: torch.dtype
    device: torch.device

class ConcreteProp(torch.fx.Interpreter):
    writer: Incomplete
    skip_offload: Incomplete
    seen_storages: Incomplete
    def __init__(self, mod, *, writer=None, skip_offload: bool = False) -> None: ...
    def run_node(self, n): ...
    pbar: Incomplete
    def propagate(self, *args): ...

def is_load_tensor_node(node): ...
def _convert_node_to_placeholder(graph, node, inps): ...
def create_minified_hlo_graph(minified_fx_graph, inputs) -> None:
    """
    Takes minified FX graph as primary input, and ports it to HLO via StableHLO
    Provides minified HLO graph as output, and archive them to local directory
    """
def dump_state(fx_g, inps) -> None: ...
def is_power_of_two(n): ...

@dataclass
class ReproState:
    graph: fx.Graph
    inps: list[torch.Tensor]
    def __post_init__(self) -> None: ...

def minifier(fail_f: fx.GraphModule, inps, module_fails, dump_state: Callable = ..., *, save_dir=None, offload_to_disk: bool = False, skip_offload: bool = False, skip_sanity: bool = False, max_granularity=None):
    """
    Minimizes a FX graph with given inputs, such that the resulting FX graph still returns True for module_fails.

    Does 2 main strategies:
    1. Truncates suffix: Removes some suffix from the graph and sets a new output.
    2. Delta Debugging: Tries replacing half of the graph with inputs. If fails,
        tries replacing quarter of the graph, etc.

    >>> # xdoctest: +SKIP(failing)
    >>> failing_function = fx.symbolic_trace(f)
    >>> minimize(failing_function, [torch.randn(5)], lambda fx_g, inps: fx_g(*inps))

    note: module_fails returns True if it fails.
    """
