from .. import graph_break_hints as graph_break_hints, variables as variables
from ..exc import raise_observed_exception as raise_observed_exception, unimplemented_v2 as unimplemented_v2
from ..utils import cmp_name_to_op_mapping as cmp_name_to_op_mapping, common_constant_types as common_constant_types, istype as istype, np as np
from .base import VariableTracker as VariableTracker
from _typeshed import Incomplete
from torch._dynamo.source import AttrSource as AttrSource, GetItemSource as GetItemSource
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator

class ConstantVariable(VariableTracker):
    """
    Variable tracker for Python literals and basic immutable types, with automatic
    routing support for collection types (lists, tuples, sets, etc.).

    The create() method intelligently constructs appropriate variable types for
    nested collections.
    """
    @staticmethod
    def create(value, **kwargs) -> VariableTracker:
        """
        Create a `ConstantVariable` based on the given value, and supports
        automatic routing for collection types like `tuple` (in which case we'd
        create `ConstantVariable` for the leaf items).

        NOTE: the caller must install the proper guards if needed; most often
        the guard will be `CONSTANT_MATCH`.
        """
    value: Incomplete
    def __init__(self, value, **kwargs) -> None: ...
    def as_proxy(self): ...
    def __repr__(self) -> str: ...
    def as_python_constant(self): ...
    def is_python_constant(self): ...
    @property
    def items(self):
        """
        Need this when adding a BaseListVariable and a ConstantVariable together.
        Happens in detectron2.
        """
    def getitem_const(self, tx: InstructionTranslator, arg: VariableTracker): ...
    @staticmethod
    def is_base_literal(obj): ...
    @staticmethod
    def is_literal(obj): ...
    def unpack_var_sequence(self, tx): ...
    def const_getattr(self, tx: InstructionTranslator, name): ...
    def call_method(self, tx: InstructionTranslator, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def call_obj_hasattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...

class EnumVariable(VariableTracker):
    """VariableTracker for enum.Enum and enum.IntEnum instances

    Provides specialized handling for Python enum types, supporting
    both standard Enum and IntEnum with proper value tracking and comparison.
    """
    value: Incomplete
    def __init__(self, value, **kwargs) -> None: ...
    @classmethod
    def create(cls, cls_type, value_vt, options): ...
    def as_proxy(self): ...
    def __repr__(self) -> str: ...
    def as_python_constant(self): ...
    def var_getattr(self, tx: InstructionTranslator, name): ...
