from typing import Callable

_MESSAGE_TEMPLATE: str

def lazy_deprecated_import(all: list[str], old_module: str, new_module: str) -> Callable:
    '''Import utility to lazily import deprecated packages / modules / functional.

    The old_module and new_module are also used in the deprecation warning defined
    by the `_MESSAGE_TEMPLATE`.

    Args:
        all: The list of the functions that are imported. Generally, the module\'s
            __all__ list of the module.
        old_module: Old module location
        new_module: New module location / Migrated location

    Returns:
        Callable to assign to the `__getattr__`

    Usage:

        # In the `torch/nn/quantized/functional.py`
        from torch.nn.utils._deprecation_utils import lazy_deprecated_import
        _MIGRATED_TO = "torch.ao.nn.quantized.functional"
        __getattr__ = lazy_deprecated_import(
            all=__all__,
            old_module=__name__,
            new_module=_MIGRATED_TO)
    '''
