import sympy
from _typeshed import Incomplete
from collections.abc import Generator
from sympy.core.expr import Expr
from sympy.core.function import Application
from sympy.core.operations import LatticeOp
from typing import SupportsFloat, TypeVar
from typing_extensions import TypeVarTuple

__all__ = ['FloorDiv', 'ModularIndexing', 'Where', 'PythonMod', 'Mod', 'CleanDiv', 'CeilToInt', 'FloorToInt', 'CeilDiv', 'IntTrueDiv', 'FloatTrueDiv', 'LShift', 'RShift', 'IsNonOverlappingAndDenseIndicator', 'TruncToFloat', 'TruncToInt', 'RoundToInt', 'RoundDecimal', 'ToFloat', 'FloatPow', 'PowByNatural', 'Identity']

_T = TypeVar('_T', bound=SupportsFloat)
_Ts = TypeVarTuple('_Ts')

class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)

    NB: This is Python-style floor division, round to -Inf
    """
    nargs: tuple[int, ...]
    precedence: int
    is_integer: bool
    @property
    def base(self) -> sympy.Basic: ...
    @property
    def divisor(self) -> sympy.Basic: ...
    def _sympystr(self, printer: sympy.printing.StrPrinter) -> str: ...
    @classmethod
    def eval(cls, base: sympy.Integer, divisor: sympy.Integer) -> sympy.Basic | None: ...
    def _ccode(self, printer): ...

class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c where % is the C modulus
    """
    nargs: tuple[int, ...]
    is_integer: bool
    precedence: int
    @classmethod
    def eval(cls, base: sympy.Integer, divisor: sympy.Integer, modulus: sympy.Integer) -> sympy.Basic | None: ...
    def _eval_is_nonnegative(self) -> bool | None: ...

class Where(sympy.Function):
    """
    Good ol' ternary operator
    """
    nargs: tuple[int, ...]
    precedence: int
    def _eval_is_integer(self) -> bool | None: ...
    def _eval_is_nonnegative(self) -> bool | None: ...
    def _eval_is_positive(self) -> bool | None: ...
    @classmethod
    def eval(cls, c: sympy.Basic, p: sympy.Basic, q: sympy.Basic) -> sympy.Basic | None: ...

class PythonMod(sympy.Function):
    nargs: tuple[int, ...]
    precedence: int
    is_integer: bool
    @classmethod
    def eval(cls, p: sympy.Expr, q: sympy.Expr) -> sympy.Expr | None: ...
    def _eval_is_nonnegative(self) -> bool | None: ...
    def _eval_is_nonpositive(self) -> bool | None: ...

class Mod(sympy.Function):
    nargs: Incomplete
    precedence: int
    is_integer: bool
    is_nonnegative: bool
    @classmethod
    def eval(cls, p, q): ...

class CleanDiv(FloorDiv):
    """
    Div where we can assume no rounding.
    This is to enable future optimizations.
    """

class CeilToInt(sympy.Function):
    is_integer: bool
    @classmethod
    def eval(cls, number): ...
    def _ccode(self, printer): ...

class FloorToInt(sympy.Function):
    is_integer: bool
    @classmethod
    def eval(cls, number): ...

class CeilDiv(sympy.Function):
    """
    Div used in indexing that rounds up.
    """
    is_integer: bool
    def __new__(cls, base, divisor): ...

class LShift(sympy.Function):
    is_integer: bool
    @classmethod
    def eval(cls, base, shift): ...

class RShift(sympy.Function):
    is_integer: bool
    @classmethod
    def eval(cls, base, shift): ...

class MinMaxBase(Expr, LatticeOp):
    def __new__(cls, *original_args, **assumptions): ...
    @classmethod
    def _satisfy_unique_summations_symbols(cls, args) -> set[sympy.core.symbol.Symbol] | None:
        '''
        One common case in some models is building expressions of the form
        max(max(max(a+b...), c+d), e+f) which is simplified to max(a+b, c+d, e+f, ...).
        For such expressions, we call the Max constructor X times (once for each nested
        max) and the expression gets flattened.

        An expensive cost in constructing those expressions is running _collapse_arguments
        and _find_localzeros. However, those two optimizations are unnecessary when the args
        to max are all of the form a+b, c+d, ..etc where each term uses a unique set of symbols.

        This function is used to detect such properties of the expressions we are building
        and if so inform that we do not need to run those optimizations. To detect those,
        we store a property in the expression that tells that this expression is a min/max
        operation over terms that use unique symbols "unique_summations_symbols". This property
        also memoize the set of symbols used in all the terms to make it faster to detect this
        property inductively.

        When we apply max to add a new term, all we need to do is check if the new term uses
        unique symbols (with respect to existing terms and itself).
        Example:
        t = Max(a+b, c+d) ==> satisfies the property
        Max(t, h+j)       ==> h,j not in [a,b,c,d] => satisfy the property.

        The function returns None if the new expression does not satisfy the unique_summations_symbols
        property. Otherwise, it returns a new set of unique symbols.
        '''
    @classmethod
    def _unique_symbols(cls, args, initial_set: set[sympy.core.symbol.Symbol] | None = None) -> set[sympy.core.symbol.Symbol] | None:
        """
        Return seen_symbols if all atoms in all args are all unique symbols,
        else returns None. initial_set can be used to represent initial value for seen_symbols
        """
    @classmethod
    def _collapse_arguments(cls, args, **assumptions):
        """Remove redundant args.

        Examples
        ========

        >>> from sympy import Min, Max
        >>> from sympy.abc import a, b, c, d, e

        Any arg in parent that appears in any
        parent-like function in any of the flat args
        of parent can be removed from that sub-arg:

        >>> Min(a, Max(b, Min(a, c, d)))
        Min(a, Max(b, Min(c, d)))

        If the arg of parent appears in an opposite-than parent
        function in any of the flat args of parent that function
        can be replaced with the arg:

        >>> Min(a, Max(b, Min(c, d, Max(a, e))))
        Min(a, Max(b, Min(a, c, d)))
        """
    @classmethod
    def _new_args_filter(cls, arg_sequence) -> Generator[Incomplete, Incomplete]:
        """
        Generator filtering args.

        first standard filter, for cls.zero and cls.identity.
        Also reshape ``Max(a, Max(b, c))`` to ``Max(a, b, c)``,
        and check arguments for comparability
        """
    @classmethod
    def _find_localzeros(cls, values, **options):
        """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.

        Unlike the sympy implementation, we only look for zero and one, we don't
        do generic is connected test pairwise which is slow
        """
    _eval_is_algebraic: Incomplete
    _eval_is_antihermitian: Incomplete
    _eval_is_commutative: Incomplete
    _eval_is_complex: Incomplete
    _eval_is_composite: Incomplete
    _eval_is_even: Incomplete
    _eval_is_finite: Incomplete
    _eval_is_hermitian: Incomplete
    _eval_is_imaginary: Incomplete
    _eval_is_infinite: Incomplete
    _eval_is_integer: Incomplete
    _eval_is_irrational: Incomplete
    _eval_is_negative: Incomplete
    _eval_is_noninteger: Incomplete
    _eval_is_nonnegative: Incomplete
    _eval_is_nonpositive: Incomplete
    _eval_is_nonzero: Incomplete
    _eval_is_odd: Incomplete
    _eval_is_polar: Incomplete
    _eval_is_positive: Incomplete
    _eval_is_prime: Incomplete
    _eval_is_rational: Incomplete
    _eval_is_real: Incomplete
    _eval_is_extended_real: Incomplete
    _eval_is_transcendental: Incomplete
    _eval_is_zero: Incomplete

class Max(MinMaxBase, Application):
    """
    Return, if possible, the maximum value of the list.
    """
    zero: Incomplete
    identity: Incomplete
    def _eval_is_positive(self): ...
    def _eval_is_nonnegative(self): ...
    def _eval_is_negative(self): ...

class Min(MinMaxBase, Application):
    """
    Return, if possible, the minimum value of the list.
    """
    zero: Incomplete
    identity: Incomplete
    def _eval_is_positive(self): ...
    def _eval_is_nonnegative(self): ...
    def _eval_is_negative(self): ...

class PowByNatural(sympy.Function):
    is_integer: bool
    precedence: int
    @classmethod
    def eval(cls, base, exp): ...

class FloatPow(sympy.Function):
    is_real: bool
    precedence: int
    @classmethod
    def eval(cls, base, exp): ...

class FloatTrueDiv(sympy.Function):
    is_real: bool
    precedence: int
    @classmethod
    def eval(cls, base, divisor): ...

class IntTrueDiv(sympy.Function):
    is_real: bool
    precedence: int
    @classmethod
    def eval(cls, base, divisor): ...
    def _ccode(self, printer): ...

class IsNonOverlappingAndDenseIndicator(sympy.Function):
    is_integer: bool
    @classmethod
    def eval(cls, *args): ...

class TruncToFloat(sympy.Function):
    is_real: bool
    @classmethod
    def eval(cls, number): ...

class TruncToInt(sympy.Function):
    is_integer: bool
    @classmethod
    def eval(cls, number): ...

class RoundToInt(sympy.Function):
    is_integer: bool
    @classmethod
    def eval(cls, number): ...

class RoundDecimal(sympy.Function):
    is_real: bool
    @classmethod
    def eval(cls, number, ndigits): ...

class ToFloat(sympy.Function):
    is_real: bool
    @classmethod
    def eval(cls, number): ...

class Identity(sympy.Function):
    """
    Prevents expansion and other optimizations
    """
    precedence: int
    def __repr__(self) -> str: ...
    def _eval_is_real(self): ...
    def _eval_is_integer(self): ...
    def _eval_expand_identity(self, **hints): ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
