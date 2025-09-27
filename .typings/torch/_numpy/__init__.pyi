from ._dtypes import *
from ._funcs import *
from ._ufuncs import *
from . import fft as fft, linalg as linalg, random as random
from ._getlimits import finfo as finfo, iinfo as iinfo
from ._ndarray import array as array, asarray as asarray, ascontiguousarray as ascontiguousarray, can_cast as can_cast, from_dlpack as from_dlpack, ndarray as ndarray, newaxis as newaxis, result_type as result_type
from ._util import AxisError as AxisError, UFuncTypeError as UFuncTypeError
from _typeshed import Incomplete
from math import e as e, pi as pi

all = all
alltrue = all
any = any
sometrue = any
inf: Incomplete
nan: Incomplete
False_: bool
True_: bool
