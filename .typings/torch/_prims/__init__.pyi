from _typeshed import Incomplete
from enum import Enum
from torch._prims_common import DimsSequenceType, RETURN_TYPE as RETURN_TYPE, TensorLikeType

__all__ = ['RETURN_TYPE', 'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh', 'bessel_i0', 'bessel_i0e', 'bessel_i1', 'bessel_i1e', 'bessel_j0', 'bessel_j1', 'bitwise_not', 'cbrt', 'ceil', 'conj_physical', 'digamma', 'erf', 'erf_inv', 'erfc', 'erfcx', 'exp', 'expm1', 'exp2', 'fill', 'floor', 'imag', 'isfinite', 'lgamma', 'log', 'log1p', 'log2', 'log10', 'ndtri', 'neg', 'real', 'reciprocal', 'round', 'sign', 'signbit', 'sin', 'sinh', 'spherical_bessel_j0', 'sqrt', 'tan', 'tanh', 'trunc', 'add', 'atan2', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'div', 'eq', 'fmax', 'fmin', 'fmod', 'frexp', 'gcd', 'ge', 'gt', 'hypot', 'igamma', 'igammac', 'le', 'lt', 'maximum', 'minimum', 'mul', 'ne', 'nextafter', 'pow', 'remainder', 'rsqrt', 'shift_left', 'shift_right_arithmetic', 'shift_right_logical', 'sub', 'zeta', 'as_strided', 'broadcast_in_dim', 'collapse_view', 'conj', 'expand_dims', 'slice', 'slice_in_dim', 'split_dim', 'squeeze', 'transpose', 'view_of', 'view_element_type', 'as_strided_scatter', 'collapse', 'cat', 'reshape', 'rev', 'where', 'clone', 'convert_element_type', 'device_put', 'item', 'maximum_value', 'minimum_value', 'copy_strided', 'copy_to', 'resize', 'amax', 'amin', 'prod', 'sum', 'xor_sum', 'var', 'empty_strided', 'empty_permuted', 'scalar_tensor', 'iota', 'svd', 'normal', '_uniform_helper', 'fft_r2c', 'fft_c2c', 'fft_c2r', '_make_token', '_sink_tokens']

class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)

abs: Incomplete
acos: Incomplete
acosh: Incomplete
asin: Incomplete
asinh: Incomplete
atan: Incomplete
atanh: Incomplete
cos: Incomplete
cosh: Incomplete
bessel_j0: Incomplete
bessel_j1: Incomplete
bessel_i0: Incomplete
bessel_i0e: Incomplete
bessel_i1: Incomplete
bessel_i1e: Incomplete
bitwise_not: Incomplete
cbrt: Incomplete
ceil: Incomplete
conj_physical: Incomplete
clone: Incomplete
digamma: Incomplete
erf: Incomplete
erf_inv: Incomplete
erfc: Incomplete
erfcx: Incomplete
exp: Incomplete
expm1: Incomplete
exp2: Incomplete
fill: Incomplete
floor: Incomplete
imag: Incomplete
isfinite: Incomplete
lgamma: Incomplete
log: Incomplete
log1p: Incomplete
log2: Incomplete
log10: Incomplete
real: Incomplete
reciprocal: Incomplete
ndtri: Incomplete
neg: Incomplete
round: Incomplete
rsqrt: Incomplete
sign: Incomplete
signbit: Incomplete
sin: Incomplete
sinh: Incomplete
spherical_bessel_j0: Incomplete
sqrt: Incomplete
tan: Incomplete
tanh: Incomplete
trunc: Incomplete
add: Incomplete
atan2: Incomplete
bitwise_and: Incomplete
bitwise_or: Incomplete
bitwise_xor: Incomplete
div: Incomplete
eq: Incomplete
fmax: Incomplete
fmin: Incomplete
fmod: Incomplete
gcd: Incomplete
ge: Incomplete
gt: Incomplete
hypot: Incomplete
igamma: Incomplete
igammac: Incomplete
le: Incomplete
lt: Incomplete
maximum: Incomplete
minimum: Incomplete
mul: Incomplete
ne: Incomplete
nextafter: Incomplete
pow: Incomplete
remainder: Incomplete
shift_left: Incomplete
shift_right_arithmetic: Incomplete
shift_right_logical = _not_impl
sub: Incomplete
zeta: Incomplete
as_strided: Incomplete
broadcast_in_dim: Incomplete
collapse_view: Incomplete
conj: Incomplete

def expand_dims(a: TensorLikeType, dimensions: DimsSequenceType, ndim=None) -> TensorLikeType:
    """
    Creates a view of a with a.ndim + len(dimensions) dimensions, with new
    dimensions of length one at the dimensions specified by dimensions.
    """

split_dim: Incomplete
squeeze: Incomplete
transpose: Incomplete
view_of: Incomplete
view_element_type: Incomplete
as_strided_scatter: Incomplete
collapse: Incomplete
cat: Incomplete
reshape: Incomplete
rev: Incomplete
where: Incomplete
convert_element_type: Incomplete
device_put: Incomplete
item: Incomplete
maximum_value: Incomplete
minimum_value: Incomplete
copy_to: Incomplete
copy_strided: Incomplete
resize: Incomplete
sum: Incomplete
xor_sum: Incomplete
prod: Incomplete
var: Incomplete
amax: Incomplete
amin: Incomplete
iota: Incomplete
empty_strided: Incomplete
empty_permuted: Incomplete
scalar_tensor: Incomplete
svd: Incomplete
normal: Incomplete
_uniform_helper: Incomplete
fft_r2c: Incomplete
fft_c2c: Incomplete
fft_c2r: Incomplete
frexp: Incomplete
_make_token: Incomplete
_sink_tokens: Incomplete

# Names in __all__ with no definition:
#   slice
#   slice_in_dim
