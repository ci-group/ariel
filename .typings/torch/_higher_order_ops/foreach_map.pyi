from _typeshed import Incomplete
from torch._higher_order_ops.base_hop import BaseHOP as BaseHOP, FunctionWithNoFreeVars as FunctionWithNoFreeVars
from typing import Any, Callable

class ForeachMap(BaseHOP):
    def __init__(self) -> None: ...
    def __call__(self, fn, *operands, **kwargs): ...

_foreach_map: Incomplete

def foreach_map(op: Callable, *operands: Any, **kwargs: dict[str, Any]): ...
