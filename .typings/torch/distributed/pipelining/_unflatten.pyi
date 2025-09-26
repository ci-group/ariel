import torch
from torch.export.unflatten import _ModuleFrame as _ModuleFrame, _SubmoduleEntry as _SubmoduleEntry

def _outline_submodules(orig_graph: torch.fx.Graph) -> torch.fx.GraphModule: ...
