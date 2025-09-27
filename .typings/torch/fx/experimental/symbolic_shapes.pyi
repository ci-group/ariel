import abc
import inspect
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
import traceback
import types
from _typeshed import Incomplete
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import _GeneratorContextManager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from torch import SymBool, SymFloat, SymInt, Tensor
from torch._dynamo.source import TensorPropertySource
from torch._guards import SLoc, ShapeGuard, Source
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.recording import ShapeEnvEvent
from torch.fx.experimental.sym_node import SymNode
from torch.types import BoolLikeType, FloatLikeType, IntLikeType
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import CppPrinter, PythonPrinter
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._traceback import CapturedTraceback
from typing import Any, Callable, Generic, NamedTuple, TypeVar
from typing_extensions import ParamSpec, TypeAlias, TypeGuard

__all__ = ['guard_or_false', 'guard_or_true', 'has_symbolic_sizes_strides', 'create_contiguous', 'ShapeEnv', 'is_concrete_int', 'is_concrete_float', 'is_concrete_bool', 'has_static_value', 'guard_int', 'guard_float', 'guard_scalar', 'canonicalize_bool_expr', 'hint_int', 'SYMPY_INTERP', 'free_symbols', 'is_symbol_binding_fx_node', 'is_nested_int', 'SHAPEENV_EVENT_KEY', 'CURRENT_NODE_KEY', 'has_free_symbols', 'has_free_unbacked_symbols', 'sym_and', 'sym_eq', 'sym_or', 'SymbolicContext', 'StatelessSymbolicContext', 'StatefulSymbolicContext', 'SubclassSymbolicContext', 'SymIntSymbolicContext', 'TrackedFake', 'statically_known_true', 'statically_known_false', 'guard_size_oblivious', 'check_consistent', 'compute_unbacked_bindings', 'ConvertIntKey', 'rebind_unbacked', 'resolve_unbacked_bindings', 'is_accessor_node', 'ValueRangesSLoc', 'SymIntEqByExpr', 'Specialization']

InputList = list
DimList = list

class GuardOnDataDependentSymNode(RuntimeError):
    cond: sympy.Basic
    def __init__(self, cond: sympy.Basic, *args: Any) -> None: ...

class PendingUnbackedSymbolNotFound(RuntimeError): ...

SHAPEENV_EVENT_KEY: str
CURRENT_NODE_KEY: str
_T = TypeVar('_T')
_SympyT = TypeVar('_SympyT', sympy.Expr, SympyBoolean, sympy.Basic)

class SymIntEqByExpr:
    """
    This is a wrapper around SymInt which has alternative semantics for
    equality.  Specifically, instead of erroring or guarding, we
    instead will hash/compare equality based on the underlying sympy
    expression; e.g., s0 and s1 will always compare as False.

    NB: This does NOT do fancy analysis that maybe_evaluate_static does;
    we can only reason through equalities that occur because to expressions
    canonicalize to the same expression via regular simplification.
    """
    val: torch.SymInt | int
    def __init__(self, val: torch.SymInt | int) -> None: ...
    def __repr__(self) -> str: ...
    def _extract(self) -> sympy.Expr: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class ConstraintViolationError(RuntimeError): ...

def has_symbolic_sizes_strides(elem: torch.Tensor) -> bool: ...
Int: TypeAlias = torch.SymInt | int

def create_contiguous(shape: Sequence[Int]) -> list[Int]: ...
def hint_int(a: torch.SymInt | int, fallback: int | None = None) -> int:
    """
    Retrieve the hint for an int (based on the underlying real values as observed
    at runtime).  If no hint is available (e.g., because data dependent shapes),
    if fallback is not None, use that instead (otherwise raise an error).
    """
Scalar: TypeAlias = torch.SymInt | torch.SymFloat | torch.SymBool | int | float | bool

def is_concrete_int(a: IntLikeType) -> bool:
    """
    Utility to check if underlying object
    in SymInt is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymInt or int): Object to test if it int
    """
def is_concrete_float(a: FloatLikeType) -> bool:
    """Utility to check if underlying object
    in SymInt is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymInt or float): Object to test if it float
    """
def is_concrete_bool(a: BoolLikeType) -> bool:
    """
    Utility to check if underlying object
    in SymBool is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymBool or bool): Object to test if it bool
    """
def has_static_value(a: SymBool | SymFloat | SymInt | bool | float | int) -> bool:
    """
    User-code friendly utility to check if a value is static or dynamic.
    Returns true if given a constant, or a symbolic expression with a fixed value.

    Args:
        a (Union[SymBool, SymFloat, SymInt, bool, float, int]): Object to test
    """
def guard_size_oblivious(expr: torch.SymBool | bool) -> bool:
    """
    Perform a guard on a symbolic boolean expression in a size oblivious way.
    This is typically used when a non-oblivious test would result in a guard
    on a data dependent value of which we don't know the value of at compile time.
    When a guard is tested this way, we may diverge in behavior from how regular
    PyTorch semantics would treat it.  For more information, see
    https://github.com/pytorch/pytorch/pull/118579
    """
def check_consistent(new: _T, old: _T) -> None:
    '''
    Test that two "meta" values (typically either Tensor or SymInt) have
    the same values, e.g., after retracing.  If we don\'t understand the
    quantities in question, we\'ll just skip the consistency check.
    '''
def resolve_unbacked_bindings(shape_env: ShapeEnv | None, bindings: dict[sympy.Symbol, pytree.KeyPath] | None) -> dict[sympy.Symbol, pytree.KeyPath] | None:
    '''
    When we do fake tensor prop, we oftentimes will allocate new unbacked symints.
    We then run proxy tensor mode, which populates node.meta["unbacked_bindings"]
    with these new symints. To ensure consistency we use PropagateUnbackedSymInts
    to rename unbacked bindings to their old ones. But all of the node metas are
    still using the old bindings from before the renaming. This function helps to
    post facto apply any renamings discovered in the PropogateUnbackedSymInts pass.
    '''
Result: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]

def rebind_unbacked(shape_env: ShapeEnv | None, n: torch.fx.Node, result: Result) -> None:
    """
    Suppose we are retracing a pre-existing FX graph that previously had
    fake tensor propagation (and therefore unbacked SymInts).  When we retrace,
    we re-propagate fake tensors, which results in new unbacked SymInts.
    When this happens, we need to tell the shape environment about the equivalence
    of the old and new unbacked SymInts.  Pass us the old torch.fx.Node (which
    has the old binding information) and the new result (which we can extract the
    new unbacked SymInts out from).
    """
def is_accessor_node(node: torch.fx.Node) -> bool:
    """
    Helper function to determine if a node is trying to access
    a symbolic integer such as size, stride, offset or item. Currently
    primarily only used in a DCE pass to figure out purity.
    """
def canonicalize_bool_expr(expr: _T) -> _T:
    """
    Canonicalize a boolean expression by transforming it into a lt / le
    inequality and moving all the non-constant terms to the rhs.
    We canonicalize And / Ors / Not via cnf and then canonicalize their subexpr
    recursively
    nb. sympy.Rel.canonical is not good enough https://github.com/sympy/sympy/issues/25924

    Args:
        expr (sympy.Expr): Expression to canonicalize
    """
def is_nested_int(s: IntLikeType) -> TypeGuard[SymInt]: ...
IterateExprs: TypeAlias = IterateExprsAtom | Sequence[IterateExprsAtom]

def free_symbols(val: IterateExprs) -> OrderedSet[sympy.Symbol]:
    """
    Recursively collect all free symbols from a value.

    This function traverses various data structures (tensors, lists, tuples, etc.) and extracts
    all sympy symbols contained within them. It's useful for finding all symbolic variables
    that a complex nested structure depends on.

    Args:
        val: The value to extract symbols from. Can be a symbolic type (SymInt, SymFloat, SymBool),
             a container (tuple, list), a tensor, or None.

    Returns:
        OrderedSet[sympy.Symbol]: An ordered set of all free symbols found in the value.
    """
def has_free_symbols(val: IterateExprs) -> bool:
    """Faster version of bool(free_symbols(val))"""
def has_free_unbacked_symbols(x: IterateExprs) -> bool:
    """Faster version of bool(free_unbacked_symbols(val))"""
def is_symbol_binding_fx_node(node: torch.fx.Node) -> sympy.Symbol | None:
    """
    Check if a given FX node is a symbol binding node.

    A symbol binding node is one that has a SymInt value in its meta that contains
    a sympy Symbol expression, and is either a placeholder node or contains unbacked symbols.

    Args:
        node (torch.fx.Node): The FX node to check

    Returns:
        Optional[sympy.Symbol]: The sympy Symbol if the node is a symbol binding node, None otherwise
    """

@dataclass(frozen=True)
class Specialization:
    """
    This class is used in multi-graph compilation contexts where we generate
    multiple specialized graphs and dispatch to the appropriate one at runtime.
    This allows us to optimize the trade-off between performance and generality
    by creating specialized versions for common patterns (e.g., x.shape[0] % 16 == 0)
    while maintaining a general fallback.
    """
    source: TensorPropertySource
    check_fn: Callable

@dataclass(frozen=True)
class ConvertIntKey:
    def __str__(self) -> str: ...
    def get(self, b: bool) -> IntLikeType:
        """Get the int value from bool"""

@dataclass(frozen=True)
class CallMethodKey:
    name: str
    def __str__(self) -> str: ...
    def get(self, o: Any) -> Any:
        """Call the method on object"""

@dataclass(frozen=True)
class InnerTensorKey:
    inner_name: str
    def __str__(self) -> str: ...
    def get(self, o: Any) -> Any:
        """Get the inner tensor attribute"""

@dataclass(frozen=True)
class DivideByKey:
    divisor: IntLikeType
    def __str__(self) -> str: ...
    def get(self, o: int) -> int:
        """Divide object by divisor"""

def compute_unbacked_bindings(shape_env: ShapeEnv | None, example_value: object, old_example_value: object | None = None, peek: bool = False) -> dict[sympy.Symbol, pytree.KeyPath] | None:
    """
    After having run fake tensor propagation and producing example_value
    result, traverse example_value looking for freshly bound unbacked
    symbols and record their paths for later.  It is an error if
    we have allocated an unbacked SymInt but it cannot be found in
    example_value.  (NB: this means if you have a multi-output
    function, you must call this on the tuple of tensor output, you
    cannot wait!)

    The peek parameter lets you check out what the bindings are without
    changing the affected list.  This is primarily useful for ensuring
    unbacked_var_to_val is promptly populated when propagate_real_tensors is on.
    """
def guard_or_false(a: BoolLikeType) -> bool:
    """
    Try to guard a, if data dependent error encountered just return false.
    """
def guard_or_true(a: BoolLikeType) -> bool:
    """
    Try to guard a, if data dependent error encountered just return true.
    """
def statically_known_false(x: BoolLikeType) -> bool:
    """
    Returns True if x can be simplified to a constant and is False.
    If x cannot be evaluated from static, we return False

    .. note::
        This function doesn't introduce new guards, so the expression may end
        up evaluating to False at runtime even if this function returns False.

    Args:
        x (bool, SymBool): The expression to try statically evaluating
    """
def statically_known_true(x: BoolLikeType) -> bool:
    """
    Returns True if x can be simplified to a constant and is true.

    .. note::
        This function doesn't introduce new guards, so the expression may end
        up evaluating to true at runtime even if this function returns False.

    Args:
        x (bool, SymBool): The expression to try statically evaluating
    """
def sym_and(x: BoolLikeType, *others: BoolLikeType) -> BoolLikeType:
    """
    and, but for symbolic expressions, without bool casting.
    """
def sym_eq(x: _T, y: _T) -> BoolLikeType:
    """
    Like ==, but when run on list/tuple, it will recursively test equality
    and use sym_and to join the results together, without guarding.
    """
def sym_or(x: BoolLikeType, *others: BoolLikeType) -> BoolLikeType:
    """
    or, but for symbolic expressions, without bool casting.
    """
def guard_scalar(a: SymBool | SymInt | SymFloat | int | bool | float) -> bool | int | float:
    """
    Guard a scalar value, which can be a symbolic or concrete boolean, integer, or float.

    This function dispatches to the appropriate guard function based on the type of the input.

    Args:
        a: A symbolic or concrete scalar value (bool, int, or float)

    Returns:
        The concrete value after guarding

    Raises:
        AssertionError: If the input is not a recognized scalar type
    """
def guard_int(a: IntLikeType) -> int: ...
def guard_float(a: FloatLikeType) -> float: ...

class DimDynamic(Enum):
    """
    Controls how to perform symbol allocation for a dimension.  It is always
    sound to default this to DYNAMIC, but the policies DUCK and STATIC can
    result in better trace-time and compile-time performance, as they reduce
    the number of allocated symbols and generally make your graph more static.

    NB: If we notice you've applied a constraint to the dimension, we will
    force it to DYNAMIC for simplicity.

    DimDynamic is controlled by a variety of higher level UX features.
    Currently:

    - In eager mode, the default policy is DUCK.
        - The default is changed to STATIC with assume_static_by_default.
        - An individual dim is marked DYNAMIC if you mark_dynamic_dim.
    - In export mode, the default policy is STATIC.
        - An individual dim is marked DYNAMIC if you specify it in
          dynamic_shapes passed to export.
    """
    DYNAMIC = 0
    DUCK = 1
    STATIC = 2
    SIZE_LIKE_UNBACKED = 3
    INFER_STRIDE = 4
    OBLIVIOUS_SIZE = 5

@dataclass(frozen=True)
class Constraint:
    warn_only: bool

@dataclass(frozen=True)
class StrictMinMaxConstraint(Constraint):
    '''
    For clients: the size at this dimension must be within \'vr\' (which
    specifies a lower and upper bound, inclusive-inclusive) AND it
    must be non-negative and should not be 0 or 1 (but see NB below).

    For backends: there must not be any guards on this dimension which
    are not implied by the given lower and upper bound.  Regardless of
    the lower bound, the backend can assume the size is non-negative
    and that it is not 0 or 1.

    An unbounded StrictMinMaxConstraint can be thought of as a strict version
    of "RelaxedUnspecConstraint".

    NB: Export will often unsoundly assume that a graph works for 0/1, even
    though at trace time we assumed size is not 0 or 1.  The idea is that
    if we produce a graph that works for a range of values, it will be OK
    for N=0/1 too.
    '''
    vr: ValueRanges
    def render(self, source: Source) -> str:
        """Format the constrain equation"""

@dataclass(frozen=True)
class RelaxedUnspecConstraint(Constraint):
    '''
    For clients: no explicit constraint; constraint is whatever is implicitly
    inferred by guards from tracing.

    For backends: there must exist at least TWO possible values for the
    size at this dimension which satisfy the guards for this dimension.

    In other words, this constraint helps us distinguish between "we don\'t
    care if this dimension specializes or not" versus "this dimension must be
    unspecialized."  However, this constraint doesn\'t say very much about what
    specialization is permitted; for example, if we guard on a size being
    even, this would still be acceptable under an unspec constraint.  This
    makes RelaxedUnspecConstraint useful for eager mode, where your backend compiler
    may add constraints to otherwise dynamic dimensions; we can\'t assert that
    there are NO guards as this is brittle because compilers should be able to
    add extra constraints.  If you want to assert that there are no guards,
    use StrictMinMaxConstraint with an unbounded ValueRanges.
    '''
    def render(self, source: Source) -> str: ...
DimConstraint = StrictMinMaxConstraint | RelaxedUnspecConstraint | None

@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    '''
    Represent and decide various kinds of equality constraints between input sources.

    A "source pair" is a pair of input sources for dynamic dimensions that
    are specified equal. We represent `source_pairs` in a union-find forest
    so that we can efficiently check whether two such sources are transitively equal.

    A "derived equality" relates an input source to an expression over a root.
    The root can be another input source, corresponding to some dynamic dimension,
    or a phantom symbol that does not directly represent any dynamic dimension. We
    represent `derived_equalities` involving input sources in a transitively-closed map
    so that we can efficiently check whether an input source is transitively equal to
    a given expression over another input source.
    (NOTE: In contrast, it is easy to decide whether an input source is transitively equal
    to a given expression over a phantom symbol; such expressions are already in canonical
    form and so the problem reduces to symbolic expression equality.)
    '''
    source_pairs: list[tuple[Source, Source]]
    derived_equalities: list[tuple[Source, Source | sympy.Symbol, Callable[[sympy.Expr], sympy.Expr]]]
    phantom_symbols: list[sympy.Symbol]
    relaxed_sources: set[Source]
    _parents: dict[Source, Source] = field(init=False)
    _defs: dict[Source, sympy.Expr] = field(init=False)
    def __post_init__(self) -> None:
        """
        Pre-processing to answer queries `is_equal` and `is_derived` below.

        Example: Suppose we are given:
          source_pairs [a = b, b = c]
          derived_equalities [d = c + 1, e = d - 1]
        We first construct a union find with source_pairs:
          _parents = {a: a, b: a, c: a}
        Then we compute canonical symbolic expressions, recursively applying derived_equalities
        until we bottom out:
          _defs = {d: c + 1, e: (c + 1) - 1 aka c}
        """
    def _find(self, source: Source) -> Source: ...
    def _union(self, root1: Source, root2: Source) -> None: ...
    def _rewrite(self, src: Source) -> sympy.Expr: ...
    def is_equal(self, source1: Source, source2: Source) -> bool: ...
    def is_derived(self, src: Source, symbol_src: Source, fn: Callable[[sympy.Expr], sympy.Expr]) -> bool: ...

@dataclass(frozen=True)
class SymbolicContext:
    '''
    Data structure specifying how we should create symbols in
    ``create_symbolic_sizes_strides_storage_offset``; e.g., should
    they be static or dynamic.

    This is an abstract base class because we are probably going to add
    another version of this that says "use exactly these SymInts, don\'t
    allocate fresh symbols."
    '''

@dataclass(frozen=True)
class SymIntSymbolicContext(SymbolicContext):
    """
    Data structure specifying any constraints on a SymInt input
    """
    constraint: DimConstraint
_P1 = ParamSpec('_P1')
_T1 = TypeVar('_T1')

@dataclass(frozen=True)
class StatelessSymbolicContext(SymbolicContext, Generic[_P1, _T1]):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by ``DimDynamic`` and ``DimConstraint``.
    This will cause fresh symbols to be allocated
    """
    dynamic_sizes: DimList[DimDynamic]
    dynamic_strides: DimList[DimDynamic] = ...
    constraint_sizes: DimList[DimConstraint] = ...
    constraint_strides: DimList[DimConstraint] = ...
    specialize_on: list[list[Callable[_P1, _T1]]] | None = ...
    view_base_context: SymbolicContext | None = ...
    def __post_init__(self) -> None: ...

@dataclass(frozen=True)
class StatefulSymbolicContext(StatelessSymbolicContext):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by a cache of Source:Symbol. A cache hit
    will reuse a stored symbol, and a cache miss will write to this cache.

    This behaves like StatelessSymbolicContext, except the cache supersedes the
    other values - dynamic_sizes and constraint_sizes will not be read if we cache
    hit.

    It is the cache owner's responsibility to maintain the lifecycle of the cache
    with respect to different shape_envs, clearing, etc.
    """
    tensor_source: Source = ...
    shape_env_to_source_to_symbol_cache: dict[int, dict[str, sympy.Expr]] = ...
    def __post_init__(self) -> None: ...

@dataclass(frozen=True)
class SubclassSymbolicContext(StatefulSymbolicContext):
    """
    The correct symbolic context for a given inner tensor of a traceable tensor subclass
    may differ from that of the outer symbolic context. This structure allows for this
    flexibility, with inner symbolic contexts mapped via attr -> symbolic context.
    """
    inner_contexts: dict[str, SymbolicContext] = ...
    def __post_init__(self) -> None: ...

@dataclass
class TrackedFake:
    """
    Tracks the sources of all fake tensors we wrap in Dynamo.
    Used by shape guard computation.
    """
    fake: FakeTensor | SymInt
    source: Source
    symbolic_context: SymbolicContext | None
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class _SymbolInfo(NamedTuple):
    k: sympy.Symbol
    vr: ValueRanges | None
    val: sympy.Integer | None
    is_size_like: bool

SYMPY_INTERP: Incomplete

@dataclass(frozen=True)
class RuntimeAssert:
    """
    This is pretty similar to ShapeGuard but it also comes with a message,
    and is exclusively used for things that MUST be true (unlike guards,
    which can evaluate False, in which case you just choose not to use
    a particular specialization)
    """
    expr: SympyBoolean
    msg: str = field(repr=False)
    stack: CapturedTraceback = field(repr=False)

class SymExprPrinter(PythonPrinter):
    def _print_Float(self, expr: sympy.Float) -> str: ...

class _ShapeGuardPrinter(abc.ABC, metaclass=abc.ABCMeta):
    """
    Abstract base class for printers that convert symbolic expressions to string representations.

    This class provides common functionality for printing symbolic expressions with
    special handling for symbols that represent tensor shapes, strides, etc.
    Subclasses implement specific formatting for different output languages.

    Args:
        symbol_to_source: Mapping from sympy symbols to their source objects
        source_ref: Function to convert a source to its string representation
        var_to_sources: Mapping from sympy symbols to their source objects (for error reporting)
    """
    symbol_to_source: Incomplete
    source_ref: Incomplete
    var_to_sources: Incomplete
    def __init__(self, symbol_to_source: Mapping[sympy.Symbol, list[Source]], source_ref: Callable[[Source], str], var_to_sources: Mapping[sympy.Symbol, list[Source]]) -> None: ...
    def _print_Float(self, expr: sympy.Float) -> str:
        """Convert a sympy Float to a Python float string representation."""
    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        """
        Convert a sympy Symbol to its source representation.

        This method looks up the symbol in symbol_to_source mapping and returns
        the string representation of its first source.

        Args:
            expr: The sympy Symbol to convert

        Returns:
            String representation of the symbol's source

        Raises:
            AssertionError: If the symbol is not found in symbol_to_source
        """
    @abc.abstractmethod
    def print_source(self, source: Source) -> str:
        """
        Convert a source object to its string representation.

        Args:
            source: The source object to convert

        Returns:
            String representation of the source
        """
    @abc.abstractmethod
    def doprint(self, expr: sympy.Expr) -> str:
        """
        Convert a sympy expression to its string representation.

        Args:
            expr: The sympy expression to convert

        Returns:
            String representation of the expression
        """

class ShapeGuardPythonPrinter(_ShapeGuardPrinter, PythonPrinter):
    """
    Python printer for shape guards that extends the base ShapeGuardPrinter.

    This class provides functionality to print symbolic expressions as Python code,
    with caching to improve performance when printing the same expressions multiple times.
    It handles printing of sources and expressions according to Python syntax.

    Args:
        *args: Arguments passed to the parent classes.
    """
    _print_cache: dict[sympy.Expr, str]
    def __init__(self, *args: Any) -> None: ...
    def print_source(self, source: Source) -> str:
        """
        Convert a source object to its string representation using the source_ref function.

        Args:
            source: The source object to convert

        Returns:
            String representation of the source
        """
    def doprint(self, expr: sympy.Expr) -> str:
        """
        Convert a sympy expression to its Python string representation with caching.

        This method first checks if the expression is already in the cache.
        If found, it returns the cached result; otherwise, it delegates to
        PythonPrinter's doprint method and caches the result.

        Args:
            expr: The sympy expression to convert

        Returns:
            String representation of the expression in Python syntax
        """

class ShapeGuardPrinter(ShapeGuardPythonPrinter): ...

class _ShapeGuardCppPrinter(_ShapeGuardPrinter, CppPrinter):
    all_symbols: set[str]
    source_to_symbol: dict[Source, sympy.Symbol]
    def __init__(self, *args: Any) -> None: ...
    def print_source(self, source: Source) -> str: ...
    def doprint(self, expr: sympy.Expr) -> str: ...

@dataclass(frozen=True)
class _ShapeGuardsHelper:
    exprs: list[str]

@dataclass(frozen=True)
class _CppShapeGuardsHelper(_ShapeGuardsHelper):
    source_to_symbol: dict[Source, sympy.Symbol]

class LoggingShapeGuardPrinter(ShapeGuardPythonPrinter):
    def __init__(self, var_to_sources: Mapping[sympy.Symbol, list[Source]]) -> None: ...

class DynamicDimConstraintPrinter(PythonPrinter):
    """
    Printer for dynamic dim constraints.
    - Instead of symbol s_k it prints its source t.size()[i]
    - Instead of Eq(_, _), Mod(_, _), etc. it prints _ == _, _ % _, etc.

    We use this to suggest code for specifying dynamic dim constraints.
    """
    symbol_to_source: Incomplete
    source_name_to_debug_name: Incomplete
    def __init__(self, symbol_to_source: dict[sympy.Symbol, list[Source]], source_name_to_debug_name: Mapping[str, str]) -> None: ...
    def _print_Symbol(self, expr: sympy.Symbol) -> str: ...

class DimConstraints:
    '''
    Custom solver for a system of constraints on symbolic dimensions.
    Solutions are "static" values or simplified "dynamic" constraints.
    '''
    _univariate_inequalities: dict[sympy.Symbol, set[SympyBoolean]]
    _symbols_with_equalities: set[sympy.Symbol]
    _substitutions: dict[sympy.Symbol, sympy.Integer]
    _var_to_val: Mapping[sympy.Symbol, sympy.Integer]
    _congruences: defaultdict[sympy.Symbol, set[sympy.Expr]]
    _multivariate_inequalities: set[SympyBoolean]
    _symbolic_equivalences: list[tuple[Source, sympy.Expr]]
    _static_results: set[str]
    _dynamic_results: set[str]
    _dcp: Incomplete
    _inconsistencies: list[str]
    _marked_dynamic: Incomplete
    _supported_sympy_functions: set[sympy.Function]
    def __init__(self, symbol_to_source: dict[sympy.Symbol, list[Source]], var_to_val: Mapping[sympy.Symbol, sympy.Integer], marked_dynamic: set[sympy.Symbol], source_name_to_debug_name: Mapping[str, str]) -> None: ...
    def rewrite_with_congruences(self, s: sympy.Symbol, expr: _SympyT) -> _SympyT:
        """
        Eliminate expressions of the form b // d and b % d while adding congruences of the form b % d == k.
        This leaves rational operators (in particular of the form b / d) that our inequality solver can handle.
        We solve the added congruences separately (using our congruence solver, see below).
        """
    _unsupported_sympy_functions: Incomplete
    def _enumerate_sympy_functions(self) -> None: ...
    def _has_unsupported_sympy_function(self, expr: sympy.Basic) -> bool:
        """
        Tracks list of sympy.Functions the export solver doesn't know how to handle.
        """
    def add(self, expr: SympyBoolean) -> bool:
        """Add an expression to the set of constraints.

        Return whether the expression is a trivial constraint (i.e., an obvious tautology).
        """
    def add_equality(self, source: Source, expr: sympy.Expr) -> None:
        """Add an equality constraint"""
    def _reduce_congruences(self) -> dict[sympy.Symbol, set[sympy.Expr]]: ...
    def _raise_inconsistencies(self) -> None: ...
    def solve(self) -> None:
        """Solve the system of constraint equations to find simplified constraints"""
    @classmethod
    def _is_supported_congruence(cls, congruence: sympy.Expr) -> bool: ...
    def forced_specializations(self) -> dict[str, sympy.Expr]:
        """Returns a dictionary of the names of symbols to their specialized value"""
    def _is_derived_dim(self, dim: object) -> TypeGuard[torch.export.dynamic_shapes._DerivedDim]: ...
    def _is_dim(self, dim: object) -> TypeGuard[torch.export.dynamic_shapes.Dim]: ...
    def _process_derived_dim_roots(self, results: dict[str, dict[str, Any]], name_to_dim: dict[str, Any]) -> None:
        '''
        Here we resolve 2 concerns with derived dims suggested fixes: 1) newly introduced roots,
        and 2) root swapping.

        1) Newly introduced roots appear with modulo guards, e.g. Mod(dx, 2) = 0 suggests
        dx is a derived dim equal to 2 * _dx, introducing a new root _dx. Currently the final
        suggested fixes handle this correctly, but we can get intermediate results that look like
        {"dy": {"eq": "dx + 1"}, "dx": {"eq": "2 * _dx + 1, "min": 3, "max": 15}}
        and this routine prettifies this by unifying to a single root, and making each suggestion
        either a derived dim or min/max range, not both.

        2) With suggested fixes for derived dims, roots can be swapped,
        e.g. dx, dx - 1 -> dy + 1, dy. Here we don\'t want to print out the attached name,
        since this leads to messages like "dx - 1 = Dim("dx - 1", ...)".
        Instead we evaluate the new root value, and remove results for its derivations.

        First we find all the original roots (specified in dynamic_shapes), that are found in the
        values of results (i.e. used for computing suggesting fix values). These original roots
        (suppose `dx`) are either specialized, unchanged, refined, or swapped
        (expressed as a derived dim). If any of the first 3 cases happen, we suggest `dx`\'s value
        in results, and remove suggestions for derivations of `dx`, assuming the derived relation
        is valid. If swapped, we find the new root, and use the fix to evaluate `dx`\'s new value,
        and then do the same with `dx`\'s derivations.

        Assuming the originally specified derived relations are correct is valid, because:
            1) if the relations are plain wrong (e.g. input shape = (6, 4) with spec (dx, dx - 1))
               produce_guards() will catch this and crash before hand.
            2) if the relations are numerically correct but do not match the emitted guard,
               for example:

                    def forward(self, x, y):
                        return x.reshape([-1]) + y  # guard: s0 * 2 = s1
                    inputs = (torch.randn(6, 2), torch.randn(12))
                    dx = Dim("dx", min=2, max=32)
                    dynamic_shapes={"x": (dx, 2), "y": (dx + 6, )}  # this matches values but not op

               then this leads to 2 linear equations, and a) produce_guards() is able to solve for
               the unique solution of dx = 6 and specialize, and b) the export constraint solver will
               raise an issue due to range constraints (a unique solution means not all values in a
               range satisfy a guard) and also force specializations.
        '''
    def prettify_results(self, original_signature: inspect.Signature, dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any], constraint_violation_error: object, forced_specializations: dict[str, str]) -> str:
        """Format a message for constraint violation erros"""

@dataclass(frozen=True)
class ShapeEnvSettings:
    """
    Encapsulates all shape env settings that could potentially affect
    FakeTensor dispatch. Used when creating dispatch cache keys.
    """
    allow_scalar_outputs: bool
    allow_dynamic_output_shape_ops: bool
    assume_static_by_default: bool
    specialize_zero_one: bool
    duck_shape: bool
    prefer_deferred_runtime_asserts_over_guards: bool
    allow_complex_guards_as_runtime_asserts: bool
    trace_asserts: bool

@dataclass
class ValueRangesSLoc:
    """
    Locations of the guards that triggered lower and upper bound.
    """
    lower: SLoc
    upper: SLoc

@dataclass
class _FrameLocalResult:
    loc: str | None = ...
    locals: dict[str, Any] = field(default_factory=dict)
    symbols: dict[str, str] = field(default_factory=dict)

class ShapeEnv:
    _translation_validation_enabled: Incomplete
    should_record_events: Incomplete
    check_recorded_events: Incomplete
    is_recording: bool
    tracked_fakes: Incomplete
    events: list[ShapeEnvEvent]
    fake_tensor_cache: dict[torch._subclasses.fake_tensor._DispatchCacheKey, torch._subclasses.fake_tensor._DispatchCacheEntry]
    def __init__(self, *, should_record_events: bool | None = None, tracked_fakes: list[Any] | None = None, **kwargs: Any) -> None: ...
    settings: Incomplete
    guards: list[ShapeGuard]
    axioms: dict[sympy.Expr, sympy.Expr]
    unique_ids: set[int]
    var_to_val: dict[sympy.Symbol, sympy.Integer]
    unbacked_var_to_val: dict[sympy.Symbol, sympy.Integer]
    oblivious_var_to_val: dict[sympy.Symbol, sympy.Integer]
    var_to_range: dict[sympy.Symbol, ValueRanges]
    var_to_range_sloc: dict[sympy.Symbol, ValueRangesSLoc]
    source_name_to_debug_name: dict[str, str]
    var_to_sources: dict[sympy.Symbol, list[Source]]
    var_to_stack: dict[sympy.Symbol, CapturedTraceback]
    source_to_var: dict[str, sympy.Symbol]
    replacements: dict[sympy.Symbol, sympy.Expr]
    replacements_slocs: dict[sympy.Symbol, SLoc]
    unbacked_renamings: dict[sympy.Symbol, sympy.Symbol]
    divisible: set[sympy.Expr]
    size_like: set[sympy.Symbol]
    val_to_var: dict[int, sympy.Symbol]
    unbacked_symfloat_counter: Incomplete
    unbacked_symint_counter: Incomplete
    deferred_runtime_asserts: dict[sympy.Symbol | None, list[RuntimeAssert]]
    num_deferred_runtime_asserts: int
    log: Incomplete
    frozen: bool
    runtime_asserts_frozen: bool
    dim_constraints: DimConstraints | None
    counter: Counter[str]
    symbol_guard_counter: Counter[sympy.Symbol]
    co_fields: Incomplete
    pending_fresh_unbacked_symbols: list[sympy.Symbol]
    _prev_cache_key: Incomplete
    _version_counter: int
    _resimplify_floor_div_axioms: bool
    fx_node_cache: dict[tuple[Callable, tuple[Any, ...]], torch.fx.Node]
    source_to_symbol: dict[str, sympy.Symbol]
    unbacked_alloc_order: dict[sympy.Symbol, int]
    user_specialization_stacks: dict[Source, traceback.StackSummary]
    framework_specialization_stacks: dict[Source, traceback.StackSummary]
    trace_asserts: Incomplete
    specializations: OrderedSet[Specialization]
    validator: Incomplete
    graph: Incomplete
    name_to_node: dict[str, torch.fx.Node]
    def _init(self, *, allow_scalar_outputs: bool = True, allow_dynamic_output_shape_ops: bool = True, assume_static_by_default: bool = False, specialize_zero_one: bool = True, duck_shape: bool | None = None, co_fields: dict[str, str] | None = None, prefer_deferred_runtime_asserts_over_guards: bool = False, allow_complex_guards_as_runtime_asserts: bool = False, trace_asserts: bool = False) -> None: ...
    @property
    def allow_scalar_outputs(self) -> bool: ...
    @property
    def allow_dynamic_output_shape_ops(self) -> bool: ...
    @property
    def assume_static_by_default(self) -> bool: ...
    @property
    def specialize_zero_one(self) -> bool: ...
    @property
    def duck_shape(self) -> bool: ...
    @property
    def prefer_deferred_runtime_asserts_over_guards(self) -> bool: ...
    @property
    def allow_complex_guards_as_runtime_asserts(self) -> bool: ...
    @contextmanager
    def patch_source_specialization(self, source: Source, check_fn: Callable[[sympy.Symbol], sympy.Expr]) -> Iterator[None]:
        '''
        Temporarily add symbol-level axioms to the ShapeEnv. This is useful when you want to "fork"
        and have parallel universes of ShapeEnvs. For example, we use this when doing multi-graph
        compile so we can support various graphs with varying levels of specializations.

        This context manager allows for temporarily adding constraints to the shape environment
        based on a specialization function applied to a symbol associated with a source.

        Args:
            source: The source of the symbol to specialize
            check_fn: A function that takes a sympy Symbol and returns a sympy expression
                     representing a constraint/specialization to be applied
        '''
    def check_equal(self, other: ShapeEnv) -> None:
        """Compare another ShapeEnv for equivalence"""
    def _snapshot_tracked_fakes(self) -> list[Any] | None: ...
    def _last_event_index(self) -> int: ...
    @contextmanager
    def _recording(self) -> Iterator[None]: ...
    def _eliminate_unbacked(self, orig_s: sympy.Symbol, new_s: sympy.Expr) -> None: ...
    def set_unbacked_var_to_val(self, k: sympy.Symbol, v: int) -> None:
        """Used only when propagate_real_tensors; registers a value for an
        unbacked symbol, which can be used last resort to resolve hints."""
    def _rename_unbacked_to(self, orig_s: sympy.Symbol, new_s: sympy.Symbol) -> None: ...
    def _constrain_is_bounded(self, a: sympy.Symbol, upper_bound: int) -> None: ...
    def _constrain_range_for_size(self, a: sympy.Symbol, min: int | None = None, max: int | None = None) -> None: ...
    def _constrain_range(self, a: sympy.Expr, min: int, max: int) -> None: ...
    def _constrain_unify(self, a: SymInt, b: SymInt) -> None:
        """
        Given two SymInts, constrain them so that they must be equal.  NB:
        this will not work with SymInts that represent nontrivial expressions
        (yet!)
        """
    def _ignore_fresh_unbacked_symbols_tls(self) -> bool: ...
    def _ignore_fresh_unbacked_symbols_set(self, b: bool) -> bool: ...
    @contextmanager
    def ignore_fresh_unbacked_symbols(self) -> Iterator[None]:
        """
        Indicates that the newly allocated unbacked SymInts are being
        discarded
        """
    def freeze(self) -> None:
        """Freeze this ShapeEnv to stop accumulating guards

        A frozen ShapeEnv will ignore any further guards generated on it and
        only emit a warning which may lead to accuracy problems.
        """
    def freeze_runtime_asserts(self) -> None:
        """Freeze this ShapeEnv to stop adding deferred runtime asserts.

        We will error if you try to install a new runtime assert when it is
        frozen.  This would indicate a lowering violation, or perhaps something
        we know statically is already True but we are checking it again in a way
        that is not clearly dischargeable.
        """
    def _create_symbol_for_source(self, source: Source) -> sympy.Symbol | None: ...
    def _add_z3var(self, symbol: sympy.Symbol, type: type) -> None: ...
    def _add_target_expr(self, expr: SympyBoolean) -> None: ...
    def _add_assertion(self, expr: SympyBoolean) -> None: ...
    def _check_translation_validate(self) -> None: ...
    def _create_fx_call_function(self, op: Callable, args: tuple) -> tuple[torch.fx.Node | None, bool]: ...
    def _create_fx_placeholder_and_z3var(self, symbol: sympy.Symbol, type: type) -> torch.fx.Node | None: ...
    def _remove_fx_node(self, node: torch.fx.Node | None) -> None: ...
    def _add_fx_node_metadata(self, node: torch.fx.Node) -> None: ...
    @staticmethod
    def _suppress_guards_tls() -> bool: ...
    def _suppress_guards_enter(self) -> None: ...
    def _suppress_guards_exit(self) -> None: ...
    def suppress_guards(self) -> _GeneratorContextManager[None]:
        """Context manager to ignore all guards generated inside"""
    def _get_key(self) -> tuple[int, int, int, int]:
        '''
        Defines the current "state" of the guards we\'ve accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        '''
    def _update_version_counter(self) -> None: ...
    def _produce_dyn_sizes(self, ex_size: Sequence[IntLikeType], source: Source, symbolic_context: SymbolicContext) -> list[sympy.Expr]: ...
    def _produce_dyn_sizes_from_int_tuple(self, tensor_size: Sequence[IntLikeType], source: Source, symbolic_context: SymbolicContext) -> list[sympy.Expr]: ...
    def create_symbolic_sizes_strides_storage_offset(self, ex: torch.Tensor, source: Source, *, symbolic_context: SymbolicContext | None = None) -> tuple[tuple[IntLikeType, ...], tuple[IntLikeType, ...], IntLikeType]:
        """
        Returns a list of symbolic sizes and strides for the given tensor.
        We try our best to express stride in terms of the sizes, so as to not
        introduce new symbolic variables.
        """
    def _maybe_specialize_sym_int_with_hint(self, maybe_sym: IntLikeType) -> IntLikeType: ...
    def _create_symbolic_sizes_strides_storage_offset(self, ex_size: Sequence[IntLikeType], ex_stride: Sequence[IntLikeType], ex_storage_offset: IntLikeType, is_dim_dynamic: Sequence[bool], source: Source, *, symbolic_context: SymbolicContext | None = None) -> tuple[tuple[IntLikeType, ...], tuple[IntLikeType, ...], IntLikeType]: ...
    def _compute_symbolic_stride(self, source: Source, size: Sequence[sympy.Expr], ex_size: Sequence[IntLikeType], ex_stride: Sequence[IntLikeType], dynamic_strides: Sequence[DimDynamic], constraint_strides: Sequence[StrictMinMaxConstraint | RelaxedUnspecConstraint | None], are_sizes_static: bool, symbolic_context: SymbolicContext) -> list[sympy.Expr]: ...
    def create_symintnode(self, sym: sympy.Expr, *, hint: int | None, source: Source | None = None) -> IntLikeType:
        """Create a SymInt value from a symbolic expression

        If you know what the current hint value of the SymInt to be created
        is, pass it into hint.  Otherwise, pass None and we will make our best
        guess

        """
    def create_symfloatnode(self, sym: sympy.Expr, *, hint: int | None, source: Source | None = None) -> FloatLikeType:
        """Create a SymFloat value from a symbolic expression"""
    def create_unspecified_symint_and_symbol(self, value: int, source: Source, dynamic_dim: DimDynamic) -> IntLikeType:
        """Create a SymInt wrapping a new unspecified symbol"""
    def create_symboolnode(self, sym: sympy.Expr) -> SymBool:
        """Create a SymBool object from a sympy boolean expression"""
    def _log_create_unbacked_symbol(self, prefix: str, symbol: sympy.Symbol, vr: ValueRanges, source: Source | None = None, sym_node: SymNode | None = None) -> None: ...
    def create_unbacked_symfloat(self) -> SymFloat:
        """Create a symbolic float without a hint value"""
    def create_unbacked_symint(self, source: Source | None = None) -> SymInt:
        """Create a symbolic integer without a hint value"""
    def is_unbacked_symint(self, symbol: sympy.Symbol) -> bool:
        """Check if a sympy symbol matches the naming convention for unbacked symbols"""
    def create_unbacked_symbool(self) -> SymBool:
        """Create a symbolic boolean without a hint value"""
    def create_unspecified_symbol(self, val: int | SymInt | float | SymFloat, source: Source, dynamic_dim: DimDynamic = ..., constraint_dim: DimConstraint = None, symbolic_context: StatelessSymbolicContext | None = None) -> sympy.Expr:
        """
        Create a symbol with an unspecified value

        Compared to standard symbols we do not assume the value is positive,
        nor do we specialze on zero or one values.
        """
    def create_symbol(self, val: int, source: Source, dynamic_dim: DimDynamic = ..., constraint_dim: DimConstraint = None, positive: bool | None = True, do_not_specialize_zero_one: bool = False, symbolic_context: StatelessSymbolicContext | None = None) -> sympy.Expr:
        """Create a new symbol which is tracked by this ShapeEnv"""
    def add_var_to_val(self, expr: sympy.Symbol, val: int) -> None:
        """Adds a new symbol to the symbolic environment."""
    def _debug_name(self, source: Source) -> str: ...
    def _render_range_for_constraint_violation(self, source: Source, c: StrictMinMaxConstraint | RelaxedUnspecConstraint) -> str: ...
    def produce_guards(self, *args: Any, **kwargs: Any) -> list[str]:
        """
        Like produce_guards_verbose, but only returns the non-verbose python guard expressions
        (no verbose guards produced.)
        """
    def produce_guards_verbose(self, placeholders: Sequence[FakeTensor], sources: Sequence[Source], source_ref: Callable[[Source], str] = ..., *, guards: list[ShapeGuard] | None = None, input_contexts: DimList[SymbolicContext] | None = None, equalities_inputs: EqualityConstraint | None = None, _simplified: bool = False, ignore_static: bool = True, langs: tuple[str, ...] = ('python', 'verbose_python')) -> list[_ShapeGuardsHelper]:
        """
        Generates a list of guards strings which, when evaluated in a context that
        defines tensors for all the sources, returns True or False depending
        on if the guards in the list evaluated to True or not.  Primarily used by Dynamo,
        but this is also helpful for manual testing of guards (see
        evaluate_guards_for_args)

        For convenience in testing, a source is allowed to be a str,
        in which case we will assume it is a LocalSource

        simplified lets you omit duck sizing, equality and 0/1 guards.
        This is useful for testing when you don't care about the boilerplate
        guards, and it may be helpful for user output too (be careful though;
        some equality guards are nontrivial!  It would be nice to get simplified
        output to print them too).  It's private because it's not
        intended for normal use

        Returns guards in python and python with verbose comments (verbose) by
        default.
        """
    def produce_guards_expression(self, placeholders: Sequence[SymInt | FakeTensor], *, guards: list[ShapeGuard] | None = None, ignore_static: bool = True) -> str | None:
        """
        Expected to be used with evaluate_guards_expression(). Produces the guards
        for the given placeholders and returns a string expression to be evaluated
        by evaluate_guards_expression given concrete values for the placeholders.
        """
    def evaluate_symexpr(self, code: str) -> int | float | bool:
        """
        To be used by compile_fx to evaluate symexprs
        """
    def deserialize_symexpr(self, code: str) -> SymInt | SymFloat | SymBool:
        """
        To be used by compile_fx to deserialize symexprs
        """
    def evaluate_guards_expression(self, code: str, args: Sequence[object]) -> bool:
        """
        Expected to be used with produce_guards_expression(). Evaluates an expression
        generated by produce_guards_expression for the given concrete args.
        """
    def evaluate_guards_for_args(self, placeholders: Sequence[FakeTensor], args: Sequence[Tensor], *, ignore_static: bool = True) -> bool:
        """Generate guards for a graph's placeholder values and evaluate the guards with args"""
    def get_pruned_guards(self, symints: Sequence[torch.SymInt]) -> list[ShapeGuard]:
        """
        Get a list of guards, but pruned so it only provides guards that
        reference symints from the passed in input
        """
    def bind_symbols(self, placeholders: Sequence[FakeTensor], args: Sequence[Tensor]) -> dict[sympy.Symbol, int]:
        """
        Given a paired list of placeholders (fake tensors with
        symbolic sizes) and concrete arguments (regular tensors
        with real sizes), returns a dictionary mapping each
        symbol to its real value.  So for example, if you
        have a placeholder with size (s0, s1), binding
        (2, 4) to it will give you {s0: 2, s1: 4}.  This is
        not guaranteed to bind ALL symbols in the ShapeEnv;
        we can't bind a symbol if it doesn't occur in any placeholder,
        and symbols that already have replacements won't get bindings.

        This is a little duplicative with evaluate_guards but
        it's different enough that it seemed cleanest to make
        another copy.  This assumes the guards are already checked,
        though if it's cheap we'll check for shenanigans
        """
    def get_nontrivial_guards(self) -> list[SympyBoolean]:
        """Returns a list of guard expressions that aren't statically known (i.e. not trivial)"""
    def format_guards(self, verbose: bool = False) -> str:
        """Format this shape env's guard expressions with optional traceback info if verbose"""
    def bound_sympy(self, expr: sympy.Expr, size_oblivious: bool = False) -> ValueRanges:
        """Given a sympy expression, computes a ValueRanges bound for what values it can be"""
    @_lru_cache
    def get_axioms(self, symbols: tuple[sympy.Symbol] | None = None, compute_hint: bool = False) -> tuple[SympyBoolean, ...]:
        """
        Given the symbols in an expression, it returns all the runtime asserts that have those symbols
        concatenated with all the guards.
        If symbols is None, it returns all the runtime asserts (and all the guards)
        """
    def get_implications(self, e: SympyBoolean) -> tuple[tuple[SympyBoolean, sympy.logic.boolalg.BooleanAtom], ...]:
        """Given a expression, it returns a list of predicates that follow from it"""
    @_lru_cache
    def _maybe_evaluate_static(self, expr: sympy.Basic, *, unbacked_only: bool = False, compute_hint: bool = False, size_oblivious: bool = False, axioms: tuple[SympyBoolean] | None = None, var_to_range: tuple[tuple[sympy.Symbol, ValueRanges]] | None = None) -> sympy.Basic | None:
        """
        Tries to evaluate expr without introducing guards

        If unbacked_only == True, then we only do substitutions on
        unbacked SymInts (leaving regular hinted integers alone).  This could
        result in an expression that still contains backed SymInts, which you
        could then potentially guard on.

        Use compute_hint == True if you are trying to compute a non-binding
        hint for the particular hint values of backed and unbacked SymInts,
        e.g., if s0 happens to be 3 this run, compute_hint will subsitute s0 with 3.
        """
    @_lru_cache
    def replace(self, expr: _SympyT) -> _SympyT:
        """
        Apply symbol replacements to any symbols in the given expression.
        """
    @_lru_cache
    def _update_divisible(self) -> None: ...
    @_lru_cache
    def simplify(self, expr: _SympyT, size_oblivious: bool = False) -> _SympyT:
        """Use known constraints and replacements to simplify the given expr"""
    def size_hint(self, expr: sympy.Basic, *, allow_none: bool = False) -> sympy.Basic | None:
        """
        Gets a size hint for a given expression from the underlying shapes we had.
        Does not introduce a guard, so only use this when you can guarantee that
        your code is still valid for arbitrary shapes (such as optimization decisions)
        """
    def has_hint(self, expr: sympy.Expr) -> bool: ...
    def _make_data_dependent_error(self, expr: sympy.Basic, unhinted_expr: sympy.Basic, *, size_oblivious_result: sympy.Basic | None = None, expr_sym_node_id: int | None = None) -> GuardOnDataDependentSymNode: ...
    def _update_var_to_range(self, symbol: sympy.Symbol, vr: ValueRanges, vr_sloc: ValueRangesSLoc | None = None, *, is_constraint: bool = False) -> None: ...
    def _set_replacement(self, a: sympy.Symbol, tgt: sympy.Expr, msg: str) -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = tgt`.
        """
    def _add_divisible(self, expr: sympy.Expr) -> None: ...
    @_lru_cache
    def _find(self, a: sympy.Symbol) -> sympy.Expr:
        """
        Implements a DSU-like algorithm to find the variable that represents a
        Also handles transitive non-identity replacements.

        a: b + c
        c: d
        """
    def _maybe_guard_rel(self, expr: sympy.Expr) -> None:
        """
        The relational guard is guarded to be true.  Use this information to
        simplify shapes (i.e. a == b or a % 5 == 0)
        """
    def _default_value_range(self, do_not_specialize_zero_one: bool = False) -> ValueRanges: ...
    def _default_unspecified_value_range(self) -> ValueRanges: ...
    @_lru_cache
    def _simplify_floor_div(self, expr: sympy.Expr) -> sympy.Expr: ...
    def _check_frozen(self, expr: sympy.Basic, concrete_val: sympy.Basic) -> None: ...
    def _get_user_frame(self) -> types.FrameType | None: ...
    def _get_stack_summary(self, is_debug: bool = False, framework_loc: str | None = None) -> tuple[SLoc, str]: ...
    def _get_sloc(self, framework_loc: str | None = None) -> SLoc: ...
    def _generate_unique_id(self, source_name: str) -> int: ...
    def _find_frame_locals(self) -> _FrameLocalResult:
        """
        Given the current user code frame, finds the relevant lines of code,
        values of symbolic locals, and free symbols involved.
        """
    def _log_guard(self, prefix: str, g: SympyBoolean, forcing_spec: bool) -> None: ...
    _expr_sym_node_id: int | None
    def evaluate_sym_node(self, sym_node: SymNode, size_oblivious: bool = False, fallback_value: bool | None = None) -> sympy.Basic:
        """
        Given a a SymNode, evaluates sym_node.expr, adding guards if necessary.
        """
    def _is_python_assert(self) -> bool: ...
    def _log_real_tensor_propagation(self, orig_expr: sympy.Basic, unsound_result: sympy.Basic) -> None: ...
    def evaluate_expr(self, orig_expr: sympy.Basic, hint: int | bool | float | None = None, fx_node: torch.fx.Node | None = None, size_oblivious: bool = False, fallback_value: bool | None = None, *, forcing_spec: bool = False) -> sympy.Basic:
        """
        Given an expression, evaluates it, adding guards if necessary
        When fallback_value is not None the function return fallback_value instead of failing with data dependent error.
        """
    def _inner_evaluate_expr(self, orig_expr: sympy.Basic, hint: int | bool | float | None, fx_node: torch.fx.Node | None, size_oblivious: bool, forcing_spec: bool, _suppress_guards_tls: bool, fallback_value: bool | None = None) -> sympy.Basic: ...
    def _log_suppressed_dde(self, a: SymBool, assumed_value: bool) -> None: ...
    def _evaluate_expr(self, orig_expr: sympy.Basic, hint: bool | int | float | None = None, fx_node: torch.fx.Node | None = None, size_oblivious: bool = False, fallback_value: bool | None = None, *, forcing_spec: bool = False) -> sympy.Basic: ...
    def cleanup(self) -> None:
        """
        Break reference cycles.

        This destroys the stacks. If you really want to keep them, we
        just need some way to break references on code objects.
        """
    def guard_or_defer_runtime_assert(self, orig_expr: SympyBoolean, msg: str, fx_node: torch.fx.Node | None = None) -> bool:
        """
        Adds a guard that orig_expr is True if we can or fall back to adding an assert
        that is checked at runtime.

        Args:
            orig_expr (sympy.Expr): Boolean expression to assert is true
            msg (str): Message to display on assertion failure
            fx_node (Optional, torch.fx.Node): node in ``self.graph`` corresponding
                to the expression, if applicable
        """
    def _refine_ranges(self, expr: SympyBoolean) -> None: ...
    def constrain_symbol_range(self, s: sympy.Symbol, compiler_min: int, compiler_max: int) -> None: ...

class PropagateUnbackedSymInts(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node) -> Result:
        """
        Run an FX node, propagating unbacked Symbol bindings to the new fake tensor
        """

class _PythonMsgPrinter(PythonPrinter):
    """
    Util printer that replaces sympy symbols with their source-level names
    and renders sympy relational operators (e.g., Eq, Ne, Ge, Le) inline
    (i.e., as ==, !=, >, <).
    """
    src_map: Incomplete
    def __init__(self, src_map: dict[str, list[str]]) -> None: ...
    def _print_Symbol(self, sym: sympy.Symbol) -> str: ...
