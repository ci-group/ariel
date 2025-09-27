import torch
from _typeshed import Incomplete
from torch._inductor import ir as ir
from torch._inductor.codegen.common import KernelTemplate as KernelTemplate
from torch._inductor.ir import Buffer as Buffer, Layout as Layout, get_free_symbols as get_free_symbols, get_symbolic_inputs as get_symbolic_inputs, gm_original_output_strides as gm_original_output_strides, ir_node_to_tensor as ir_node_to_tensor
from torch._inductor.runtime.benchmarking import benchmarker as benchmarker
from torch._inductor.utils import do_bench_using_profiling as do_bench_using_profiling
from torch._inductor.virtualized import V as V
from typing import Any, Callable

log: Incomplete

class SubgraphChoiceCaller(ir.ChoiceCaller):
    """
    Represents a Subgraph Autotuning choice, and the subgraph can be any arbitrary
    GraphModule. Compiles the Subgraph down to a module for benchmarking.
    """
    example_inputs: Incomplete
    gm: Incomplete
    sym_inputs: Incomplete
    def __init__(self, name: str, input_nodes: list[Buffer], layout: Layout, description: str, make_fx_graph: Callable[..., Any]) -> None: ...
    def __str__(self) -> str: ...
    def benchmark(self, *args: list[Any], out: torch.Tensor) -> float: ...
    def hash_key(self) -> str: ...
    def output_node(self) -> ir.TensorBox: ...
    def info_dict(self) -> dict[str, Any]:
        """Information returned here is logged to the autotune log file when that is enabled."""
    def autoheuristic_id(self) -> str: ...

class SubgraphTemplate(KernelTemplate):
    """
    A template for subgraph evaluation to be used in autotuning.

    This class allows creating customized subgraphs that can be appended
    as choices during the autotuning process, enabling the selection of
    optimal implementations for complex operations.
    """
    index_counter: Incomplete
    name: Incomplete
    make_fx_graph: Incomplete
    def __init__(self, name: str, make_fx_graph: Callable[..., Any]) -> None:
        """
        Initialize a subgraph template.

        Args:
            name: The name of this template
            graph: The FX graph
        """
    def generate(self, input_nodes: list[Buffer], layout: Layout, **kwargs: Any) -> SubgraphChoiceCaller:
        """
        Generate a SubgraphChoiceCaller instance for autotuning.

        Args:
            input_nodes: List of input nodes to the subgraph
            layout: Memory layout information for the output
            example_inputs: Example tensor inputs used to trace and benchmark the subgraph
            **kwargs: Additional keyword arguments

        Returns:
            SubgraphChoiceCaller: A callable object that can be used for autotuning
        """
