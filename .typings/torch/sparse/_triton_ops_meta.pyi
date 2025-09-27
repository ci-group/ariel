__all__ = ['get_meta', 'tune_bsr_dense_addmm', 'tune__int_bsr_dense_addmm']

def get_meta(op, key, device_name=None, version=..., exact: bool = False):
    '''Return triton kernel meta parameters of the specified op and its inputs key.

    Parameters
    ----------
    op (str): The name of an operation that implementation uses meta parameters.
    key (tuple): A tuple of op input parameters, e.g. shapes, etc.
    device_name (optional, str): The name of a device for which op
      parameters are provided.
    version (optional, hashable): Specifies the version of parameters.
    exact (optional, bool): When True, the returned data (if
      available) corresponds exactly to the specified device_name and
      version information. Otherwise, if the corresponding data is not
      available but there exists a data set that is computed for a
      similar GPU device, then this data set will be returned.

    Returns
    -------
    result (dict): The requested mapping of parameter names and
      values, or None when no data is available. If the input `key`
      contains `"*"`, the result will be a dictionary of keys and
      mappings that match with the given `key`.
    '''
def tune__int_bsr_dense_addmm(input, bsr, dense, *, beta: int = 1, alpha: int = 1, out=None, store: bool = False, verbose: bool = False, force: bool = False): ...
def tune_bsr_dense_addmm(input, bsr, dense, *, beta: int = 1, alpha: int = 1, left_alpha=None, right_alpha=None, out=None, store: bool = False, verbose: bool = False, force: bool = False, opname=None):
    """Tune bsr_dense_addmm kernel parameters against the given inputs.

    When store is True, the tuning results will be stored in the
    database of kernel parameters.
    """
