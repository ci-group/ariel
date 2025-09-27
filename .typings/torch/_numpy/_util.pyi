from . import _dtypes_impl as _dtypes_impl

def is_sequence(seq): ...

class AxisError(ValueError, IndexError): ...
class UFuncTypeError(TypeError, RuntimeError): ...

def cast_if_needed(tensor, dtype): ...
def cast_int_to_float(x): ...
def normalize_axis_index(ax, ndim, argname=None): ...
def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate: bool = False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.
    Used internally by multi-axis-checking logic.

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`
    """
def allow_only_single_axis(axis): ...
def expand_shape(arr_shape, axis): ...
def apply_keepdims(tensor, axis, ndim): ...
def axis_none_flatten(*tensors, axis=None):
    """Flatten the arrays if axis is None."""
def typecast_tensor(t, target_dtype, casting):
    """Dtype-cast tensor to target_dtype.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to cast
    target_dtype : torch dtype object
        The array dtype to cast all tensors to
    casting : str
        The casting mode, see `np.can_cast`

     Returns
     -------
    `torch.Tensor` of the `target_dtype` dtype

     Raises
     ------
     ValueError
        if the argument cannot be cast according to the `casting` rule

    """
def typecast_tensors(tensors, target_dtype, casting): ...
def _try_convert_to_tensor(obj): ...
def _coerce_to_tensor(obj, dtype=None, copy: bool = False, ndmin: int = 0):
    '''The core logic of the array(...) function.

    Parameters
    ----------
    obj : tensor_like
        The thing to coerce
    dtype : torch.dtype object or None
        Coerce to this torch dtype
    copy : bool
        Copy or not
    ndmin : int
        The results as least this many dimensions
    is_weak : bool
        Whether obj is a weakly typed python scalar.

    Returns
    -------
    tensor : torch.Tensor
        a tensor object with requested dtype, ndim and copy semantics.

    Notes
    -----
    This is almost a "tensor_like" coersion function. Does not handle wrapper
    ndarrays (those should be handled in the ndarray-aware layer prior to
    invoking this function).
    '''
def ndarrays_to_tensors(*inputs):
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)"""
