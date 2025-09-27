import torch
from .triton_compat import Config as Config
from collections.abc import Hashable
from torch._inductor.runtime.cache_dir_utils import cache_dir as cache_dir, default_cache_dir as default_cache_dir, triton_cache_dir as triton_cache_dir
from typing import Any

def conditional_product(*args: int) -> int: ...
def ceildiv(number: int, denom: int) -> int: ...
def is_power_of_2(n: int) -> bool:
    """Returns whether n = 2 ** m for some integer m."""
def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
def triton_config_to_hashable(cfg: Config) -> Hashable:
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
def validate_triton_config(cfg: Config) -> None: ...
def create_bandwidth_info_str(ms: float, num_gb: float, gb_per_s: float, prefix: str = '', suffix: str = '', color: bool = True) -> str: ...
def get_max_y_grid() -> int: ...

HAS_COLORAMA: bool

def _color_text(msg: str, color: str) -> str: ...
def green_text(msg: str) -> str: ...
def yellow_text(msg: str) -> str: ...
def red_text(msg: str) -> str: ...
def blue_text(msg: str) -> str: ...
def get_first_attr(obj: Any, *attrs: str) -> Any:
    """
    Return the first available attribute or throw an exception if none is present.
    """
dynamo_timed = torch._dynamo.utils.dynamo_timed

def triton_hash_to_path_key(key: str) -> str: ...
def compile_mps_shader(source: str) -> Any:
    """
    Compiles shader source but raise more actionable error message when needed
    """
