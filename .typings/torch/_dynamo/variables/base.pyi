from . import builder as builder
from .. import graph_break_hints as graph_break_hints, variables as variables
from ..codegen import PyCodegen as PyCodegen
from ..current_scope_id import current_scope_id as current_scope_id
from ..exc import raise_observed_exception as raise_observed_exception, unimplemented_v2 as unimplemented_v2
from ..guards import GuardBuilder as GuardBuilder, install_guard as install_guard
from ..source import AttrSource as AttrSource, Source as Source
from ..symbolic_convert import InstructionTranslator as InstructionTranslator, InstructionTranslatorBase as InstructionTranslatorBase
from ..utils import cmp_name_to_op_mapping as cmp_name_to_op_mapping, istype as istype
from _typeshed import Incomplete
from collections.abc import Sequence
from enum import Enum
from typing import Any, Callable

class SourceType(Enum):
    """
    This Enum divides VariableTracker into 2 cases, depending on the variable
    it represents:
    - already existed that Dynamo began tracking while introspection (Existing)
    - is a new variable that is created during Dynamo introspection (New)

    In general, we have these invariants:
    1. for `VariableTracker` associated with `Existing`, its `source` field must not be None.
    2. for `VariableTracker` associated with `New`, most of the time its
       `source` field is None, except for cases like side effect codegen for
       `AttributeMutationNew`, during which we generate a
       `LocalSource('tmp...')` for such variable, to facilitate codegen.
    """
    Existing = 0
    New = 1

class MutationType:
    """
    Base class for Variable.mutation_type. It encodes information about
    1. The type of mutation Dynamo allows on the variable.
    2. Whether the value represented by this variable already existed before
    Dynamo tracing.
    """
    scope: int
    def __init__(self, typ: SourceType) -> None: ...

class ValueMutationNew(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value itself (rather than its attributes).
    2. The value is created by the bytecode Dynamo is tracing through.

    For instance, Dynamo could model a newly created list with this marker,
    indicating that while we need to model mutations to this list, we don't have
    to emit bytecode for these mutations if the list doesn't escape into the
    Python world.
    """
    def __init__(self) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...

class ValueMutationExisting(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value itself (rather than its attributes).
    2. The value exists before Dynamo tracing started.

    For instance, Dynamo could model a pre-existing list with this marker,
    indicating that if we encounter mutations to this list, we need to buffer
    and re-apply those mutations after the graph runs, since the list might be
    used afterwards in Python.
    """
    is_modified: bool
    def __init__(self, is_modified: bool = False) -> None: ...

class AttributeMutation(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates that Dynamo
    allows mutation on the value's attributes.
    """
    def __init__(self, typ: SourceType) -> None: ...

class AttributeMutationExisting(AttributeMutation):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value's attributes.
    2. The value exists before Dynamo tracing started.

    For instance, Dynamo could model a pre-existing object with this marker,
    indicating that if we encounter mutations to this object, we need to buffer
    then re-apply those mutations after the graph runs, since the object might
    be used afterwards in Python.
    """
    def __init__(self) -> None: ...

class AttributeMutationNew(AttributeMutation):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value's attributes.
    2. The value is created by the bytecode Dynamo is tracing through.

    For instance, Dynamo could model a newly created object with this marker,
    indicating that while we need to model mutations to this object, we don't
    have to emit bytecode for these mutations if the object doesn't escape into
    the Python world.
    """
    cls_source: Incomplete
    def __init__(self, cls_source: Source | None = None) -> None: ...

def _is_top_level_scope(scope_id): ...
def is_side_effect_safe(m: MutationType): ...

class AsPythonConstantNotImplementedError(NotImplementedError):
    vt: VariableTracker
    def __init__(self, vt: VariableTracker) -> None: ...

class VariableTrackerMeta(type):
    all_subclasses: Incomplete
    def __instancecheck__(cls, instance) -> bool:
        """Make isinstance work with LazyVariableTracker"""
    def __init__(cls, name, bases, attrs) -> None: ...

class VariableTracker(metaclass=VariableTrackerMeta):
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.

    Prefer the factory function VariableTracker.build() over VariableTracker.__init__().
    """
    _nonvar_fields: Incomplete
    def clone(self, **kwargs):
        """Shallow copy with some (optional) changes"""
    @classmethod
    def visit(cls, fn: Callable[[VariableTracker], None], value: Any, cache: dict[int, Any] | None = None) -> None:
        """
        Walk value and call fn on all the VariableTracker instances
        """
    def __repr__(self) -> str: ...
    def debug_repr(self): ...
    def python_type(self):
        """
        Abstract method to be implemented by subclasses of VariableTracker.

        This method should return the type represented by the instance of the subclass.
        The purpose is to provide a standardized way to retrieve the Python type information
        of the variable being tracked.

        Returns:
            type: The Python type (such as int, str, list, etc.) of the variable tracked by
                the subclass. If the type cannot be determined or is not relevant,
                leaving it undefined or invoking super() is always sound.

        Note:
            This is an abstract method and may be overridden in subclasses.

        Example:
            class SetVariable(VariableTracker):
                def python_type(self):
                    return set

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
    def python_type_name(self): ...
    def as_python_constant(self) -> None:
        """For constants"""
    def guard_as_python_constant(self):
        """Similar to as_python_constant(), but add ID_MATCH guards to try to force things to become constants"""
    def is_python_constant(self): ...
    def make_guard(self, fn): ...
    def const_getattr(self, tx: InstructionTranslator, name: str) -> Any:
        """getattr(self, name) returning a python constant"""
    def var_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker:
        """getattr(self, name) returning a new variable"""
    def is_proxy(self): ...
    def as_proxy(self) -> None: ...
    def maybe_fx_node(self): ...
    def reconstruct(self, codegen: PyCodegen): ...
    def unpack_var_sequence(self, tx) -> list['VariableTracker']: ...
    def force_unpack_var_sequence(self, tx) -> list['VariableTracker']: ...
    def has_unpack_var_sequence(self, tx) -> bool: ...
    def has_force_unpack_var_sequence(self, tx) -> bool: ...
    def force_apply_to_var_sequence(self, tx, fn) -> None: ...
    def inspect_parameter_names(self) -> list[str]: ...
    def call_obj_hasattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...
    def call_function(self, tx: InstructionTranslator, args: Sequence['VariableTracker'], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def set_name_hint(self, name) -> None: ...
    def realize(self) -> VariableTracker:
        """Used by LazyVariableTracker to build the real VariableTracker"""
    def unwrap(self) -> VariableTracker:
        """Used by LazyVariableTracker to return the real VariableTracker if it already exists"""
    def is_realized(self):
        """Used by LazyVariableTracker to indicate an unrealized node"""
    def next_variable(self, tx) -> None: ...
    def is_strict_mode(self, tx): ...
    def is_mutable(self):
        """Whether Dynamo allows mutation on this variable."""
    def is_immutable(self):
        """Whether Dynamo bans mutation on this variable."""
    @staticmethod
    def build(tx: InstructionTranslatorBase, value: Any, source: Source | None = None) -> Any:
        """Create a new VariableTracker from a value and optional Source"""
    source: Incomplete
    mutation_type: Incomplete
    def __init__(self, *, source: Source = None, mutation_type: MutationType = None) -> None: ...

def typestr(*objs): ...
