import dataclasses
import dis
from _typeshed import Incomplete
from typing import Any

TERMINAL_OPCODES: Incomplete
JUMP_OPCODES: Incomplete
JUMP_OPNAMES: Incomplete
HASLOCAL: Incomplete
HASFREE: Incomplete
stack_effect = dis.stack_effect

def get_indexof(insts):
    """
    Get a mapping from instruction memory address to index in instruction list.
    Additionally checks that each instruction only appears once in the list.
    """
def remove_dead_code(instructions):
    """Dead code elimination"""
def remove_pointless_jumps(instructions):
    """Eliminate jumps to the next instruction"""
def propagate_line_nums(instructions) -> None:
    """Ensure every instruction has line number set in case some are removed"""
def remove_extra_line_nums(instructions) -> None:
    """Remove extra starts line properties before packing bytecode"""

@dataclasses.dataclass
class ReadsWrites:
    reads: set[Any]
    writes: set[Any]
    visited: set[Any]

def livevars_analysis(instructions, instruction): ...

@dataclasses.dataclass
class FixedPointBox:
    value: bool = ...

@dataclasses.dataclass
class StackSize:
    low: int | float
    high: int | float
    fixed_point: FixedPointBox
    def zero(self) -> None: ...
    def offset_of(self, other, n) -> None: ...
    def exn_tab_jump(self, depth) -> None: ...

def stacksize_analysis(instructions) -> int | float: ...
