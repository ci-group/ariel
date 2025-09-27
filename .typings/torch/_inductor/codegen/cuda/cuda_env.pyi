import functools
from ... import config as config
from _typeshed import Incomplete
from torch._inductor.utils import clear_on_fresh_cache as clear_on_fresh_cache

log: Incomplete

@clear_on_fresh_cache
def get_cuda_arch() -> str | None: ...
@clear_on_fresh_cache
def get_cuda_version() -> str | None: ...
@functools.cache
def nvcc_exist(nvcc_path: str | None = 'nvcc') -> bool: ...
