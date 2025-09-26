from dataclasses import dataclass
from torch.fx import GraphModule
from typing import Any, Callable

__all__ = ['reference_representation_rewrite']

@dataclass
class _RewriteInfo:
    """Data needed for rewrite, this includes example inputs, pattern and replacement functions
    and post transformation functions for the exported pattern and replacement GraphModule
    """
    example_inputs: tuple[Any, ...]
    pattern: Callable
    replacement: Callable
    pattern_post_trans: Callable[[GraphModule], GraphModule] | None = ...
    replacement_post_trans: Callable[[GraphModule], GraphModule] | None = ...

def reference_representation_rewrite(model: GraphModule) -> GraphModule: ...
