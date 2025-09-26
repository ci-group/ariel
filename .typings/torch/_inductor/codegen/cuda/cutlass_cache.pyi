import functools
from _typeshed import Incomplete
from torch._inductor.codecache import cutlass_key as cutlass_key
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch as get_cuda_arch, get_cuda_version as get_cuda_version
from torch._inductor.codegen.cuda.serialization import get_cutlass_operation_serializer as get_cutlass_operation_serializer
from torch._inductor.runtime.cache_dir_utils import cache_dir as cache_dir
from torch._inductor.utils import clear_on_fresh_cache as clear_on_fresh_cache
from typing import Any

log: Incomplete
CONFIG_PREFIX: str

def get_config_request_key(arch: str, cuda_version: str, instantiation_level: str) -> str:
    """
    Return a key for the full ops, based on cutlass key, arch, cuda version, and instantiation level.
    """
def _generate_config_filename(request_key: str) -> str:
    """
    Generate a filename for the full ops.
    """
@clear_on_fresh_cache
@functools.cache
def maybe_fetch_ops() -> list[Any] | None:
    """
    Fetch ops from databases.
    """
