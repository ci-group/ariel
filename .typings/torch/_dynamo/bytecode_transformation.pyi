import dataclasses
import dis
import functools
import types
from ..utils._backport_slots import dataclass_slots as dataclass_slots
from .bytecode_analysis import get_indexof as get_indexof, propagate_line_nums as propagate_line_nums, remove_extra_line_nums as remove_extra_line_nums, stacksize_analysis as stacksize_analysis
from .utils import is_safe_constant as is_safe_constant
from _typeshed import Incomplete
from collections.abc import Iterator, Sequence
from typing import Any, Callable

@dataclasses.dataclass
class InstructionExnTabEntry:
    start: Instruction
    end: Instruction
    target: Instruction
    depth: int
    lasti: bool
    def __repr__(self) -> str: ...
    def __eq__(self, o) -> bool: ...

@dataclasses.dataclass
class Instruction:
    """A mutable version of dis.Instruction"""
    opcode: int
    opname: str
    arg: int | None
    argval: Any
    offset: int | None = ...
    starts_line: int | None = ...
    is_jump_target: bool = ...
    positions: dis.Positions | None = ...
    target: Instruction | None = ...
    exn_tab_entry: InstructionExnTabEntry | None = ...
    argrepr: str | None = ...
    def __hash__(self) -> int: ...
    def __eq__(self, other) -> bool: ...
    def short_inst_repr(self) -> str: ...
    def copy_positions(self, other: Instruction) -> None: ...

def convert_instruction(i: dis.Instruction) -> Instruction: ...

class _NotProvided:
    def __repr__(self) -> str: ...

def inst_has_op_bits(name): ...
def create_instruction(name, *, arg=None, argval=..., target=None) -> Instruction:
    '''
    At most one of `arg`, `argval`, and `target` can be not None/_NotProvided.
    This is to prevent ambiguity, e.g. does
        create_instruction("LOAD_CONST", 5)
    mean load the constant at co_consts[5], or load the constant 5?

    If `arg` is not provided, it will be computed during assembly from
    `argval` or `target`.

    Bits in the args of instructions LOAD_GLOBAL, LOAD_ATTR (3.12+), and LOAD_SUPER_ATTR
    modify the behavior of the instruction. In this case, we allow both `arg`
    and `argval` to be set. The value of `arg` here is expected to be the value of
    the op bits and the true value of `arg` will be computed during assembly.
    If `arg` is not set, the bits are assumed to be 0.
    '''
def create_jump_absolute(target) -> Instruction: ...
def create_load_const(val, checked: bool = True) -> Instruction:
    """
    In general we should only create `LOAD_CONST` for immutable objects, but
    sometimes it's convenient _and safe_ for Dynamo create `LOAD_CONST` for
    mutable objects. In such cases, use `checked=False`.
    """
def create_dup_top() -> Instruction: ...
def create_rot_n(n) -> list[Instruction]:
    '''
    Returns a "simple" sequence of instructions that rotates TOS to the n-th
    position in the stack. For Python < 3.11, returns a single ROT_*
    instruction. If no such instruction exists, an error is raised and the
    caller is expected to generate an equivalent sequence of instructions.
    For Python >= 3.11, any rotation can be expressed as a simple sequence of
    swaps.
    '''
def add_push_null(inst_or_insts: Instruction | list[Instruction]) -> list[Instruction]:
    """
    Appends or prepends a PUSH_NULL instruction to `inst_or_insts`,
    depending on Python version. Used when you know that
    `inst_or_insts` generates a callable that will be called.

    NOTE: Assumes `inst_or_insts` is a single instruction or sequence of
    instructions that pushes exactly 1 object to the stack that is to
    be called. It is important that you include ALL instructions that
    construct the callable - not just the first instruction/a prefix.

    Will attempt to use the NULL push bit for instructions
    with such bits (LOAD_GLOBAL 3.11+, LOAD_ATTR 3.12+, LOAD_SUPER_ATTR).
    In this case, instructions WILL be modified.
    """
def add_push_null_call_function_ex(inst_or_insts: Instruction | list[Instruction]) -> list[Instruction]:
    """Like add_push_null, but the low bit of LOAD_ATTR/LOAD_SUPER_ATTR
    is not set, due to an expected CALL_FUNCTION_EX instruction.
    """
def create_call_function(nargs, push_null) -> list[Instruction]:
    '''
    Creates a sequence of instructions that makes a function call.

    `push_null` is used in Python 3.11+ only. It is used in codegen when
    a function call is intended to be made with the NULL + fn convention,
    and we know that the NULL has not been pushed yet. We will push a
    NULL and rotate it to the correct position immediately before making
    the function call.

    `push_null` should be True if no NULL is pushed for the callable.
    Conversely, `push_null` should be False if a NULL was pushed for the callable.
    Prefer using `push_null=False` when possible since we will not need to rotate
    NULL to the right place, which is less efficient.

    Generally, you should codegen a function by using `add_push_null` then
    `create_call_function` with `push_null=False`.

    Example of when to set push_null False:

    insts = [
        create_instruction("LOAD_GLOBAL", argval="torch"),
        create_instruction("LOAD_ATTR", argval="nn"),
        create_instruction("LOAD_ATTR", argval="functional"),
        create_instruction("LOAD_ATTR", argval="relu"),
    ]
    insts = add_push_null(insts)
    insts.append(create_instruction("LOAD_FAST", argval="x"))
    insts.extend(create_call_function(1, False))

    Example of when to set push_null True:

    insts = [create_instruction("LOAD_FAST", x)]
    for should_wrap, wrapper_name in wrappers:
        if should_wrap:
            insts.extend([
                create_instruction("LOAD_GLOBAL", argval="wrapper1"),
                create_instruction("SWAP", arg=2),
                *create_call_function(1, True),
            )
    '''
def create_call_method(nargs) -> list[Instruction]: ...
def create_load_method(name) -> Instruction: ...
def create_setup_with(target) -> Instruction: ...
def create_swap(n) -> list[Instruction]: ...
def lnotab_writer(lineno: int, byteno: int = 0) -> tuple[list[int], Callable[[int, int], None]]:
    """
    Used to create typing.CodeType.co_lnotab
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table if Python < 3.10
    """
def linetable_310_writer(first_lineno):
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table for Python 3.10
    """
def encode_varint(n: int) -> list[int]:
    """
    6-bit chunk encoding of an unsigned integer
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    """
def linetable_311_writer(first_lineno: int):
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    This is the internal format of the line number table for Python 3.11
    """

@dataclasses.dataclass
class ExceptionTableEntry:
    start: int
    end: int
    target: int
    depth: int
    lasti: bool

def encode_exception_table_varint(n: int) -> list[int]:
    """
    Similar to `encode_varint`, but the 6-bit chunks are ordered in reverse.
    """
def decode_exception_table_varint(bytes_iter: Iterator[int]) -> int:
    """
    Inverse of `encode_exception_table_varint`.
    """
def check_exception_table(tab: list[ExceptionTableEntry]) -> None:
    """
    Verifies that a list of ExceptionTableEntries will make a well-formed
    jump table: entries are non-empty, sorted, and do not overlap.
    """
def parse_exception_table(exntab: bytes) -> list[ExceptionTableEntry]:
    """
    Parse the exception table according to
    https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
    """
def assemble_exception_table(tab: list[ExceptionTableEntry]) -> bytes:
    """
    Inverse of parse_exception_table - encodes list of exception
    table entries into bytes.
    """
def assemble(instructions: list[Instruction], firstlineno: int) -> tuple[bytes, bytes]:
    """Do the opposite of dis.get_instructions()"""
def _get_instruction_by_offset(offset_to_inst: dict[int, Instruction], offset: int):
    """
    Get the instruction located at a given offset, accounting for EXTENDED_ARGs
    """
def virtualize_jumps(instructions) -> None:
    """Replace jump targets with pointers to make editing easier"""

_REL_JUMPS: Incomplete

def flip_jump_direction(instruction: Instruction) -> None: ...
def _get_instruction_front(instructions: list[Instruction], idx: int):
    """
    i.e. get the first EXTENDED_ARG instruction (if any) when targeting
    instructions[idx] with a jump.
    """
def devirtualize_jumps(instructions) -> None:
    """Fill in args for virtualized jump target after instructions may have moved"""
def virtualize_exception_table(exn_tab_bytes: bytes, instructions: list[Instruction]):
    """Replace exception table entries with pointers to make editing easier"""
def compute_exception_table(instructions: list[Instruction]) -> list[ExceptionTableEntry]:
    """Compute exception table in list format from instructions with exn_tab_entries"""
def check_inst_exn_tab_entries_nested(tab: list[InstructionExnTabEntry], indexof) -> None:
    '''
    Checks `tab` is a properly sorted list of nested InstructionExnTabEntry\'s,
    i.e. no entries partially overlap.
    "Properly sorted" means entries are sorted by increasing starts, then
    decreasing ends.
    '''
def propagate_inst_exn_table_entries(instructions: list[Instruction]) -> None:
    """
    Copies exception table entries to all instructions in an entry's range.
    Supports nested exception table entries.
    """
def check_inst_exn_tab_entries_valid(instructions: list[Instruction]):
    """
    Checks that exn_tab_entries of instructions are valid.
    An entry's start, end, and target must be in instructions.
    Instructions with an exn_tab_entry are located within
    the entry's start and end instructions.
    Instructions do not share exn_tab_entries.

    Implicitly checks for no duplicate instructions.
    """
def strip_extended_args(instructions: list[Instruction]) -> None: ...
def overwrite_instruction(old_inst, new_insts): ...
def remove_load_call_method(instructions: list[Instruction]) -> list[Instruction]:
    """LOAD_METHOD puts a NULL on the stack which causes issues, so remove it"""
def remove_jump_if_none(instructions: list[Instruction]) -> None: ...
def remove_binary_store_slice(instructions: list[Instruction]) -> None: ...

FUSED_INSTS: Incomplete

def remove_fused_load_store(instructions: list[Instruction]) -> None: ...
def explicit_super(code: types.CodeType, instructions: list[Instruction]) -> None:
    """convert super() with no args into explicit arg form"""
def fix_extended_args(instructions: list[Instruction]) -> int:
    """Fill in correct argvals for EXTENDED_ARG ops"""
def instruction_size(inst) -> int: ...
def check_offsets(instructions) -> None: ...
def update_offsets(instructions) -> None: ...
def debug_bytes(*args) -> str: ...
def debug_checks(code) -> None:
    """Make sure our assembler produces same bytes as we start with"""

HAS_LOCAL: Incomplete
HAS_NAME: Incomplete
HAS_FREE: Incomplete
HAS_CONST: Incomplete

def get_const_index(code_options, val) -> int: ...
def fix_vars(instructions: list[Instruction], code_options, varname_from_oparg=None): ...
def clear_instruction_args(instructions) -> None: ...
@functools.lru_cache
def get_code_keys() -> list[str]: ...
def transform_code_object(code, transformations, safe: bool = False) -> types.CodeType: ...
def clean_and_assemble_instructions(instructions: list[Instruction], keys: list[str], code_options: dict[str, Any]) -> tuple[list[Instruction], types.CodeType]: ...
def populate_kw_names_argval(instructions, consts) -> None: ...
def cleaned_instructions(code, safe: bool = False) -> list[Instruction]: ...
def _clone_instructions(instructions): ...
@functools.lru_cache
def _cached_cleaned_instructions(code, safe: bool = False) -> Sequence[Instruction]: ...

_unique_id_counter: Incomplete

def unique_id(name, with_uuid: bool = False) -> str: ...
def is_generator(code: types.CodeType) -> bool: ...
def bytecode_from_template(fn, varname_map=None, noreturn: bool = True, noprefix: bool = True):
    """Generates bytecode from a template function `fn` for use in
    dynamo bytecode generation.

    For example, we can generate Python-version-independent bytecode
    for looping through a dictionary and copying the values to a new dictionary.

    def template(d1, d2):
        for k, v in d1.items():
            d2[k] = v


    or a try block:

    def template():
        try:
            dummy1
        except:
            dummy2
            raise
        dummy3

    Args:
        fn: a function template to generate bytecode from
        varname_map: a mapping of `fn`'s varnames to new names. This
            map will be applied to the generated bytecode's varnames.
            For example, local variables in `fn` can be replaced with
            new names that are generated by `OutputGraph.new_var`.
        noreturn: remove all RETURN_* bytecodes and replace them with a jump
            to the end of the bytecode. NOTE: any items pushed to the stack
            for return WILL remain on the stack! Append a POP_TOP if you don't want
            that item to be present.
        noprefix: remove prefix bytecodes (all bytecode before the first RESUME, inclusive).
    """
