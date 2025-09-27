import sympy
from _typeshed import Incomplete
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._sympy.value_ranges import ValueRanges
from typing import NamedTuple

__all__ = ['InputDim']

class InputDim(NamedTuple):
    input_name: str
    dim: int

class _AddRuntimeAssertionsForInlineConstraintsPass(PassBase):
    range_constraints: dict[sympy.Symbol, ValueRanges]
    _asserts_generated_unbacked_symbols: set[sympy.Symbol]
    counter: int
    def __init__(self, range_constraints: dict[sympy.Symbol, ValueRanges]) -> None: ...
    def _assert_range_constraint(self, node, lower, upper, assert_msg) -> None: ...
    def _insert_assert_async(self, last_node, op, lower, upper, assert_msg):
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
    existing_inline_assertions: Incomplete
    def call(self, graph_module) -> PassResult: ...
