from ._normalizations import ArrayLike, ArrayLikeOrScalar, CastingModes, DTypeLike, NotImplementedType, OutArray, normalizer

__all__ = ['add', 'arctan2', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'copysign', 'divide', 'equal', 'float_power', 'floor_divide', 'fmax', 'fmin', 'fmod', 'gcd', 'greater', 'greater_equal', 'heaviside', 'hypot', 'lcm', 'left_shift', 'less', 'less_equal', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_or', 'logical_xor', 'maximum', 'minimum', 'mod', 'multiply', 'nextafter', 'not_equal', 'power', 'remainder', 'right_shift', 'subtract', 'true_divide', 'divmod', 'modf', 'matmul', 'ldexp', 'abs', 'absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh', 'bitwise_not', 'cbrt', 'ceil', 'conj', 'conjugate', 'cos', 'cosh', 'deg2rad', 'degrees', 'exp', 'exp2', 'expm1', 'fabs', 'fix', 'floor', 'invert', 'isfinite', 'isinf', 'isnan', 'log', 'log10', 'log1p', 'log2', 'logical_not', 'negative', 'positive', 'rad2deg', 'radians', 'reciprocal', 'rint', 'sign', 'signbit', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh', 'trunc']

@normalizer
def matmul(x1: ArrayLike, x2: ArrayLike, /, out: OutArray | None = None, *, casting: CastingModes | None = 'same_kind', order: NotImplementedType = 'K', dtype: DTypeLike | None = None, subok: NotImplementedType = False, signature: NotImplementedType = None, extobj: NotImplementedType = None, axes: NotImplementedType = None, axis: NotImplementedType = None): ...
@normalizer
def ldexp(x1: ArrayLikeOrScalar, x2: ArrayLikeOrScalar, /, out: OutArray | None = None, *, where: NotImplementedType = True, casting: CastingModes | None = 'same_kind', order: NotImplementedType = 'K', dtype: DTypeLike | None = None, subok: NotImplementedType = False, signature: NotImplementedType = None, extobj: NotImplementedType = None): ...
@normalizer
def divmod(x1: ArrayLike, x2: ArrayLike, out1: OutArray | None = None, out2: OutArray | None = None, /, out: tuple[OutArray | None, OutArray | None] = (None, None), *, where: NotImplementedType = True, casting: CastingModes | None = 'same_kind', order: NotImplementedType = 'K', dtype: DTypeLike | None = None, subok: NotImplementedType = False, signature: NotImplementedType = None, extobj: NotImplementedType = None): ...
def modf(x, /, *args, **kwds): ...

# Names in __all__ with no definition:
#   abs
#   absolute
#   add
#   arccos
#   arccosh
#   arcsin
#   arcsinh
#   arctan
#   arctan2
#   arctanh
#   bitwise_and
#   bitwise_not
#   bitwise_or
#   bitwise_xor
#   cbrt
#   ceil
#   conj
#   conjugate
#   copysign
#   cos
#   cosh
#   deg2rad
#   degrees
#   divide
#   equal
#   exp
#   exp2
#   expm1
#   fabs
#   fix
#   float_power
#   floor
#   floor_divide
#   fmax
#   fmin
#   fmod
#   gcd
#   greater
#   greater_equal
#   heaviside
#   hypot
#   invert
#   isfinite
#   isinf
#   isnan
#   lcm
#   left_shift
#   less
#   less_equal
#   log
#   log10
#   log1p
#   log2
#   logaddexp
#   logaddexp2
#   logical_and
#   logical_not
#   logical_or
#   logical_xor
#   maximum
#   minimum
#   mod
#   multiply
#   negative
#   nextafter
#   not_equal
#   positive
#   power
#   rad2deg
#   radians
#   reciprocal
#   remainder
#   right_shift
#   rint
#   sign
#   signbit
#   sin
#   sinh
#   sqrt
#   square
#   subtract
#   tan
#   tanh
#   true_divide
#   trunc
