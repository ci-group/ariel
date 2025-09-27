from _typeshed import Incomplete

__all__ = ['dtype', 'DType', 'typecodes', 'issubdtype', 'set_default_dtype', 'sctypes', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'complex64', 'complex128', 'bool_', 'intp', 'int_', 'intc', 'byte', 'short', 'longlong', 'ulonglong', 'ubyte', 'half', 'single', 'double', 'float_', 'csingle', 'singlecomplex', 'cdouble', 'cfloat', 'complex_', 'generic', 'number', 'integer', 'signedinteger', 'unsignedinteger', 'inexact', 'floating', 'complexfloating']

class generic:
    name: str
    def __new__(cls, value): ...

class number(generic):
    name: str

class integer(number):
    name: str

class inexact(number):
    name: str

class signedinteger(integer):
    name: str

class unsignedinteger(integer):
    name: str

class floating(inexact):
    name: str

class complexfloating(inexact):
    name: str

class int8(signedinteger):
    name: str
    typecode: str
    torch_dtype: Incomplete

class int16(signedinteger):
    name: str
    typecode: str
    torch_dtype: Incomplete

class int32(signedinteger):
    name: str
    typecode: str
    torch_dtype: Incomplete

class int64(signedinteger):
    name: str
    typecode: str
    torch_dtype: Incomplete

class uint8(unsignedinteger):
    name: str
    typecode: str
    torch_dtype: Incomplete

class uint16(unsignedinteger):
    name: str
    typecode: str
    torch_dtype: Incomplete

class uint32(signedinteger):
    name: str
    typecode: str
    torch_dtype: Incomplete

class uint64(signedinteger):
    name: str
    typecode: str
    torch_dtype: Incomplete

class float16(floating):
    name: str
    typecode: str
    torch_dtype: Incomplete

class float32(floating):
    name: str
    typecode: str
    torch_dtype: Incomplete

class float64(floating):
    name: str
    typecode: str
    torch_dtype: Incomplete

class complex64(complexfloating):
    name: str
    typecode: str
    torch_dtype: Incomplete

class complex128(complexfloating):
    name: str
    typecode: str
    torch_dtype: Incomplete

class bool_(generic):
    name: str
    typecode: str
    torch_dtype: Incomplete

sctypes: Incomplete

def dtype(arg): ...

class DType:
    _scalar_type: Incomplete
    def __init__(self, arg) -> None: ...
    @property
    def name(self): ...
    @property
    def type(self): ...
    @property
    def kind(self): ...
    @property
    def typecode(self): ...
    def __eq__(self, other): ...
    @property
    def torch_dtype(self): ...
    def __hash__(self): ...
    def __repr__(self) -> str: ...
    __str__ = __repr__
    @property
    def itemsize(self): ...
    def __getstate__(self): ...
    def __setstate__(self, value) -> None: ...

typecodes: Incomplete

def set_default_dtype(fp_dtype: str = 'numpy', int_dtype: str = 'numpy'):
    '''Set the (global) defaults for fp, complex, and int dtypes.

    The complex dtype is inferred from the float (fp) dtype. It has
    a width at least twice the width of the float dtype,
    i.e., it\'s complex128 for float64 and complex64 for float32.

    Parameters
    ----------
    fp_dtype
        Allowed values are "numpy", "pytorch" or dtype_like things which
        can be converted into a DType instance.
        Default is "numpy" (i.e. float64).
    int_dtype
        Allowed values are "numpy", "pytorch" or dtype_like things which
        can be converted into a DType instance.
        Default is "numpy" (i.e. int64).

    Returns
    -------
    The old default dtype state: a namedtuple with attributes ``float_dtype``,
    ``complex_dtypes`` and ``int_dtype``. These attributes store *pytorch*
    dtypes.

    Notes
    ------------
    This functions has a side effect: it sets the global state with the provided dtypes.

    The complex dtype has bit width of at least twice the width of the float
    dtype, i.e. it\'s complex128 for float64 and complex64 for float32.

    '''
def issubdtype(arg1, arg2): ...

# Names in __all__ with no definition:
#   byte
#   cdouble
#   cfloat
#   complex_
#   csingle
#   double
#   float_
#   half
#   int_
#   intc
#   intp
#   longlong
#   short
#   single
#   singlecomplex
#   ubyte
#   ulonglong
