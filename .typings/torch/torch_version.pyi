from _typeshed import Incomplete
from typing import Any

__all__ = ['TorchVersion']

class TorchVersion(str):
    """A string with magic powers to compare to both Version and iterables!
    Prior to 1.10.0 torch.__version__ was stored as a str and so many did
    comparisons against torch.__version__ as if it were a str. In order to not
    break them we have TorchVersion which masquerades as a str while also
    having the ability to compare against both packaging.version.Version as
    well as tuples of values, eg. (1, 2, 1)
    Examples:
        Comparing a TorchVersion object to a Version object
            TorchVersion('1.10.0a') > Version('1.10.0a')
        Comparing a TorchVersion object to a Tuple object
            TorchVersion('1.10.0a') > (1, 2)    # 1.2
            TorchVersion('1.10.0a') > (1, 2, 1) # 1.2.1
        Comparing a TorchVersion object against a string
            TorchVersion('1.10.0a') > '1.2'
            TorchVersion('1.10.0a') > '1.2.1'
    """
    __slots__: Incomplete
    def _convert_to_version(self, inp: Any) -> Any: ...
    def _cmp_wrapper(self, cmp: Any, method: str) -> bool: ...
