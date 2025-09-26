from _typeshed import Incomplete
from torch._prims_common import TensorLikeType

__all__ = ['bfloat16', 'bool', 'byte', 'cdouble', 'cfloat', 'chalf', 'char', 'double', 'float', 'half', 'int', 'long', 'short', 'complex', 'polar']

bfloat16: Incomplete
bool: Incomplete
byte: Incomplete
cdouble: Incomplete
cfloat: Incomplete
chalf: Incomplete
char: Incomplete
double: Incomplete
float: Incomplete
half: Incomplete
int: Incomplete
long: Incomplete
short: Incomplete

def complex(real: TensorLikeType, imag: TensorLikeType) -> TensorLikeType: ...
def polar(abs: TensorLikeType, angle: TensorLikeType) -> TensorLikeType: ...
