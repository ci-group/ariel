from torch._higher_order_ops.auto_functionalize import auto_functionalized as auto_functionalized, auto_functionalized_v2 as auto_functionalized_v2
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized as decompose_auto_functionalized
from torch.export import ExportedProgram as ExportedProgram
from torch.fx import Graph as Graph

def remove_self_clone(graph: Graph) -> None: ...
def unsafe_remove_auto_functionalized_pass(ep: ExportedProgram) -> ExportedProgram:
    """
    This pass removes an instances of the higher order op 'auto_functionalized',
    and modifies the calling EP inplace to have the original mutator op.
    This pass doesn't perform safety checks to make sure that this inplace mutation is safe.
    """
