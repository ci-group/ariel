import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult

__all__ = ['DuplicateDQPass']

class DuplicateDQPass(PassBase):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult: ...
