import dataclasses
import types
from .bytecode_transformation import Instruction as Instruction, bytecode_from_template as bytecode_from_template, create_call_function as create_call_function, create_instruction as create_instruction, create_jump_absolute as create_jump_absolute, create_load_const as create_load_const, overwrite_instruction as overwrite_instruction, transform_code_object as transform_code_object, unique_id as unique_id
from .utils import ExactWeakKeyDictionary as ExactWeakKeyDictionary
from _typeshed import Incomplete
from typing import Any

CO_OPTIMIZED: int
CO_NEWLOCALS: int
CO_VARARGS: int
CO_VARKEYWORDS: int
CO_NESTED: int
CO_GENERATOR: int
CO_NOFREE: int
CO_COROUTINE: int
CO_ITERABLE_COROUTINE: int
CO_ASYNC_GENERATOR: int
TORCH_DYNAMO_RESUME_IN_PREFIX: str

def _initial_push_null(insts) -> None: ...
def _bytecode_from_template_with_split(template, stack_index, varname_map=None): ...
def _try_except_tf_mode_template(dummy, stack_var_name) -> None: ...

@dataclasses.dataclass(frozen=True)
class ReenterWith:
    stack_index: int
    target_values: tuple[Any, ...] | None = ...
    def try_except_torch_function_mode(self, code_options, cleanup: list[Instruction]):
        """
        Codegen based off of:
        try:
            (rest)
        except:
            (restore previous tf mode stack)
            raise
        """
    def try_finally(self, code_options, cleanup: list[Instruction]):
        """
        Codegen based off of:
        load args
        enter context
        try:
            (rest)
        finally:
            exit context
        """
    def __call__(self, code_options, cleanup):
        """
        Codegen based off of:
        with ctx(args):
            (rest)
        """

@dataclasses.dataclass
class ResumeFunctionMetadata:
    code: types.CodeType
    instructions: list[Instruction] = dataclasses.field(default_factory=list)
    prefix_block_target_offset_remap: list[int] = dataclasses.field(default_factory=list)
    block_target_offset_remap: dict[int, int] | None = ...

def _filter_iter(l1, l2, cond):
    """
    Two-pointer conditional filter.
    e.g. _filter_iter(insts, sorted_offsets, lambda i, o: i.offset == o)
    returns the instructions with offsets in sorted_offsets
    """
def _load_tuple_and_call(tup): ...

class ContinueExecutionCache:
    cache: Incomplete
    generated_code_metadata: Incomplete
    @classmethod
    def lookup(cls, code, lineno, *key): ...
    @classmethod
    def generate(cls, code, lineno, offset: int, setup_fn_target_offsets: tuple[int, ...], nstack: int, argnames: tuple[str, ...], argnames_null: tuple[str, ...], setup_fns: tuple[ReenterWith, ...], stack_ctx_vars: tuple[tuple[int, tuple[Any, ...]], ...], argnames_ctx_vars: tuple[tuple[str, tuple[Any, ...]], ...], null_idxes: tuple[int, ...]) -> types.CodeType: ...
    @staticmethod
    def unreachable_codes(code_options) -> list[Instruction]:
        """Codegen a `raise None` to make analysis work for unreachable code"""
    @classmethod
    def generate_based_on_original_code_object(cls, code, lineno, offset: int, setup_fn_target_offsets: tuple[int, ...], *args):
        """
        This handles the case of generating a resume into code generated
        to resume something else.  We want to always generate starting
        from the original code object so that if control flow paths
        converge we only generated 1 resume function (rather than 2^n
        resume functions).
        """
