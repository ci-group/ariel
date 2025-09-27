import dataclasses
from . import config as config
from .utils import get_backend_num_stages as get_backend_num_stages
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Generator
from functools import partial
from threading import Lock
from torch.utils._ordered_set import OrderedSet as OrderedSet
from triton import Config as TritonConfig
from typing import Any, Callable

@dataclasses.dataclass
class BaseConfig:
    """
    Base Gemm configuration used for most backends (CPU, CUDA)
    """
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int

@dataclasses.dataclass
class GemmConfig(BaseConfig):
    """
    Gemm configuration used for most backends (CPU, CUDA)
    """
    group_m: int = ...
ConvConfig = BaseConfig

@dataclasses.dataclass
class FlexConfig:
    """
    Base Config class for flex attention
    - FlexAttn forward, backward and flex decode will use this

    NOTE:
    For flex_attn bwd block_m and block_n are reused for block_m1, block_m2, block_n1, block_n2

    """
    block_m: int
    block_n: int
    num_stages: int
    num_warps: int

@dataclasses.dataclass
class FlexDecodeConfig:
    """
    Config class for flex decoding
    """
    block_n: int
    num_stages: int
    num_warps: int

@dataclasses.dataclass
class ROCmGemmConfig(GemmConfig):
    """
    ROCm subclass for GEMMs, with AMD backend specific tuneable kernargs
    """
    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmConvConfig(ConvConfig):
    """
    ROCm subclass for Conv, with AMD backend specific tuneable kernargs
    """
    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmFlexConfig(FlexConfig):
    """
    ROCm subclass for FlexAttn, with AMD backend specific tuneable kernargs
    """
    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmFlexDecodeConfig(FlexDecodeConfig):
    """
    ROCm subclass for FlexDecode, with AMD backend specific tuneable kernargs
    """
    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

class BaseHeuristicSingleton(type):
    """
    Thread-safe implementation of single to be used in the config heuristic subclasses
    to ensure heavy __init__ calls are not repeatedly run
    """
    _instances: dict[type[Any], Any]
    _lock: Lock
    def __call__(cls, *args: Any, **kwargs: Any) -> BaseConfigHeuristic: ...

class BaseConfigHeuristic(metaclass=BaseHeuristicSingleton):
    """
    Base class for mm_configs, device specific triton kernels config inherit from here
    """
    mm_configs: list[BaseConfig]
    exhaustive_configs: list[BaseConfig]
    extra_mm_configs: list[BaseConfig]
    int8_mm_configs: list[BaseConfig]
    mixed_mm_configs: list[BaseConfig]
    persistent_mm_configs: list[BaseConfig]
    scaled_mm_configs: list[BaseConfig]
    scaled_persistent_mm_configs: list[BaseConfig]
    mm_plus_mm_configs: list[BaseConfig]
    conv_configs: list[BaseConfig]
    flex_attn_fwd_autotune_configs: list[FlexConfig]
    flex_attn_bwd_autotune_configs: list[FlexConfig]
    flex_decode_autotune_configs: list[FlexDecodeConfig]
    exhaustive_flex_attn_fwd_configs: list[FlexConfig]
    exhaustive_flex_attn_bwd_configs: list[FlexConfig]
    exhaustive_flex_decode_configs: list[FlexDecodeConfig]
    def __init__(self) -> None: ...
    def _finalize_mm_configs(self, configs: list[BaseConfig]) -> Generator[TritonConfig, None, None]:
        """
        Finalizes configs after scaling, applying additional constraints.
        """
    def _scale_mm_configs(self, m: int, n: int, k: int, configs: list[BaseConfig], scale: float, has_int8_tensor: bool, exclude: Callable[[int, int, int], bool]) -> list[BaseConfig]:
        """
        Scales and filters matrix multiplication configs based on input size.
        """
    def _prune_exhaustive_configs(self, configs: list[BaseConfig], dtype_size: int) -> list[BaseConfig]: ...
    def preprocess_mm_configs(self, m: int, n: int, k: int, configs: list[BaseConfig], has_int8_tensor: bool = False, scale: int = 1, exclude: Callable[[int, int, int], bool] = ..., dtype_size: int = 0) -> Generator[TritonConfig, None, None]: ...
    def triton_config(self, num_stages: int, num_warps: int, **kwargs: Any) -> TritonConfig: ...
    def get_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_exhaustive_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_extra_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_int8_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_mixed_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_persistent_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_scaled_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_scaled_persistent_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_mm_plus_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_conv_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_decode_configs(self, head_dim: int, dtype: Any) -> list[FlexDecodeConfig]: ...

class CPUConfigHeuristic(BaseConfigHeuristic): ...

class CUDAConfigHeuristic(BaseConfigHeuristic):
    """
    Child class for CUDA device specific gemm/flex attention/conv/ configs.
    """
    h100_default_flex_config: Incomplete
    a100_default_flex_config: Incomplete
    def __init__(self) -> None: ...
    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_decode_configs(self, head_dim: int, dtype: Any) -> list[FlexDecodeConfig]: ...

class ROCmConfigHeuristic(BaseConfigHeuristic):
    """
    Child class for ROCm specific gemm/flex attention/conv/ configs.
    """
    default_num_stages: Incomplete
    mm_configs: list[BaseConfig]
    exhaustive_configs: list[BaseConfig]
    default_flex_config: Incomplete
    flex_attn_fwd_autotune_configs: list[FlexConfig]
    flex_attn_bwd_autotune_configs: list[FlexConfig]
    flex_decode_autotune_configs: list[FlexDecodeConfig]
    exhaustive_flex_attn_fwd_configs: list[FlexConfig]
    exhaustive_flex_attn_bwd_configs: list[FlexConfig]
    exhaustive_flex_decode_configs: list[FlexDecodeConfig]
    def __init__(self) -> None: ...
    def _filter_configs(self, configs: list[BaseConfig], new_num_stages: int) -> list[BaseConfig]: ...
    def _finalize_mm_configs(self, configs: list[BaseConfig]) -> Generator[TritonConfig, None, None]:
        """
        Finalizes configs after scaling, applying additional constraints.
        """
    def get_extra_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_int8_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_mixed_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_persistent_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_scaled_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_scaled_persistent_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_mm_plus_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_conv_configs(self) -> partial[Generator[TritonConfig, None, None]]: ...
    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_decode_configs(self, head_dim: int, dtype: Any) -> list[FlexDecodeConfig]: ...

class XPUConfigHeuristic(BaseConfigHeuristic):
    """
    Placeholder child class for XPU specific overrides.
    """
