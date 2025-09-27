import torch
from _typeshed import Incomplete
from types import TracebackType

class _InsertPoint:
    insert_point: Incomplete
    g: Incomplete
    guard: Incomplete
    def __init__(self, insert_point_graph: torch._C.Graph, insert_point: torch._C.Node | torch._C.Block) -> None: ...
    prev_insert_point: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None: ...

def insert_point_guard(self, insert_point: torch._C.Node | torch._C.Block) -> _InsertPoint: ...
