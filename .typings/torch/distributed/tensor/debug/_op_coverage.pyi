import torch.fx
import torch.nn as nn
from _typeshed import Incomplete
from torch._functorch.compilers import aot_module as aot_module
from torch._inductor.decomposition import select_decomp_table as select_decomp_table
from torch.distributed.tensor import DTensor as DTensor

inductor_decomps: Incomplete
graphs: list[torch.fx.GraphModule]

def fwd_bwd_compiler(fx_g, _): ...
def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    """
    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.

    Convenient util to get the fwd and bwd graphs of an arbitrary model
    with inductor decompositions. Note that this would simply do tracing
    with aot_module and don't ensure correctness. This is useful to track
    the ops needed in DTensor.
    """
def print_op_coverage_summary(model: nn.Module, args, kwargs, *, output_csv: bool = False):
    """
    Util to print the operator coverage summary of a certain model with tabulute.

    Must have tabulate module installed.
    """
