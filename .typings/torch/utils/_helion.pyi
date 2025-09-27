import functools
from torch.utils._triton import has_triton as has_triton

@functools.cache
def has_helion_package() -> bool: ...
@functools.cache
def has_helion() -> bool: ...
