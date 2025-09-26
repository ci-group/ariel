import torch
from torch.fx.passes.infra.pass_base import PassBase as PassBase, PassResult as PassResult

class _RemoveRuntimeAssertionsPass(PassBase):
    """
    Remove runtime assertions inserted by the
    _AddRuntimeAssertionsForInlineConstraintsPass.
    """
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult: ...
