from torch.fx.node import Node
from typing import Any, Callable

__all__ = ['map_arg', 'map_aggregate']

def map_arg(a: Any, fn: Callable[[Node], Any]) -> Any: ...
def map_aggregate(a: Any, fn: Callable[[Any], Any]) -> Any: ...
