import functools
import sympy
from .functions import BitwiseFn_bitwise_and as BitwiseFn_bitwise_and, BitwiseFn_bitwise_or as BitwiseFn_bitwise_or, CeilToInt as CeilToInt, CleanDiv as CleanDiv, FloatPow as FloatPow, FloatTrueDiv as FloatTrueDiv, FloorDiv as FloorDiv, FloorToInt as FloorToInt, Identity as Identity, IntTrueDiv as IntTrueDiv, IsNonOverlappingAndDenseIndicator as IsNonOverlappingAndDenseIndicator, Max as Max, Min as Min, Mod as Mod, ModularIndexing as ModularIndexing, OpaqueUnaryFn_log2 as OpaqueUnaryFn_log2, PowByNatural as PowByNatural, PythonMod as PythonMod, RoundDecimal as RoundDecimal, RoundToInt as RoundToInt, ToFloat as ToFloat, TruncToFloat as TruncToFloat, TruncToInt as TruncToInt, Where as Where
from _typeshed import Incomplete
from sympy.logic.boolalg import Boolean as SympyBoolean
from typing import Any

log: Incomplete

@functools.cache
def handlers(): ...

ASSOCIATIVE_OPS: Incomplete

def _run_sympy_handler(analysis, args, expr, index_dtype=...): ...

_nil: Incomplete

def sympy_interp(analysis, env: dict[sympy.Symbol, Any], expr: sympy.Expr | SympyBoolean, *, index_dtype=..., missing_handler=None): ...
