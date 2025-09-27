import sympy
from .loop_body import LoopBody as LoopBody
from .utils import dominated_nodes as dominated_nodes
from torch.utils._sympy.value_ranges import ValueRanges as ValueRanges
from typing import Any

def val_expressable_in_32_bits(val: Any) -> bool: ...
def range_expressable_in_32_bits(range: ValueRanges[sympy.Expr]) -> bool: ...
def try_to_reduce_precision(node: Any, bounds: dict[Any, Any], indirect_vars: list[Any], indices: dict[Any, sympy.Expr], replacement_vals: dict[Any, ValueRanges[sympy.Expr]]) -> None: ...
def indexing_dtype_strength_reduction(loop_body: LoopBody) -> None:
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32
    """
