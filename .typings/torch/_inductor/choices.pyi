import sympy
import torch
import typing
from . import config as config
from .codecache import write_text as write_text
from .codegen.simd_kernel_features import SIMDKernelFeatures as SIMDKernelFeatures
from .codegen.triton import TritonKernel as TritonKernel
from .metrics import get_metric_table as get_metric_table, is_metric_table_enabled as is_metric_table_enabled
from .runtime.hints import DeviceProperties as DeviceProperties, ReductionHint as ReductionHint
from .scheduler import BaseSchedulerNode as BaseSchedulerNode, Scheduler as Scheduler, WhyNoFuse as WhyNoFuse
from .template_heuristics import BaseConfigHeuristic as BaseConfigHeuristic, CPUConfigHeuristic as CPUConfigHeuristic, CUDAConfigHeuristic as CUDAConfigHeuristic, ROCmConfigHeuristic as ROCmConfigHeuristic, XPUConfigHeuristic as XPUConfigHeuristic
from .virtualized import V as V
from collections.abc import Generator
from functools import partial
from torch.utils._ordered_set import OrderedSet as OrderedSet
from triton import Config as TritonConfig
from typing import Any

class Sortable(typing.Protocol):
    """Anything that can be used as a list.sort() key (int/tuple/etc)"""
    def __lt__(self, other: typing.Self) -> bool: ...

class InductorChoices:
    """
    This class contains a collection of default heuristics that effect performance of our generated
    code.  We try to not put correctness requirements in this file.

    You can override the choices made here by doing:

            class MyHeuristics(InductorChoices):
                ...

            torch._inductor.virtualized.V.set_choices_handler(MyHeuristics())
    """
    def get_config_heuristics(self, device_type: str | None = 'cuda') -> BaseConfigHeuristic: ...
    def get_base_mm_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_extra_mm_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_int8_mm_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_mixed_mm_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_persistent_mm_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_scaled_mm_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_scaled_persistent_mm_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_mm_plus_mm_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_conv_configs(self, device_type: str | None = 'cuda') -> partial[Generator[TritonConfig, None, None]]: ...
    def get_flex_attention_fwd_configs(self, head_dim: int, dtype: torch.dtype, device_type: str | None = 'cuda') -> list[Any]: ...
    def get_flex_attention_bwd_configs(self, head_dim: int, dtype: torch.dtype, device_type: str | None = 'cuda') -> list[Any]: ...
    def get_flex_decode_configs(self, head_dim: int, dtype: torch.dtype, device_type: str | None = 'cuda') -> list[Any]: ...
    def triton_kernel_kwargs(self, kernel_cls: type[TritonKernel], features: SIMDKernelFeatures, groups: list[sympy.Expr], kernel_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Hook to change the kwargs passed to TritonKernel, used to apply fixed configurations"""
    @staticmethod
    def should_use_cooperative_reduction(features: SIMDKernelFeatures) -> bool:
        """Heuristic to decide if a cooperative reduction should be used."""
    @staticmethod
    def should_use_persistent_reduction(features: SIMDKernelFeatures, cooperative_reduction: bool) -> bool:
        """
        Heuristic to decide if a persistent reduction should be used.
        """
    @staticmethod
    def want_no_x_dim(features: SIMDKernelFeatures) -> bool:
        """
        Heuristic to decide if we should drop the X dimension from a persistent reduction kernel.
        So the [XBLOCK, RBLOCK] block becomes a [RBLOCK] block and XBLOCK is forced to be always 1.
        Strangely this is faster than a [1, RBLOCK] block in some cases.
        """
    @staticmethod
    def reduction_split_factor(device: torch.device, reduction_numel_hint: int, numel_hint: int, inner_reduction: bool) -> int:
        """Heuristic to decide the RSPLIT used for split reductions.
        When a reduction has a small number of outputs there is not enough parallelism,
        so we will do the reduction in two phases."""
    @staticmethod
    def can_fuse(scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode, shared_data_score: int) -> bool:
        """
        Heuristics to prevent fusion applied to both horizontal and vertical fusions.  Heuristics here should not
        be needed for correctness and tweaking them may yield additional performance.

        See also some related heuristics that can be changed via config:
            - config.triton.tiling_prevents_pointwise_fusion
            - config.triton.tiling_prevents_reduction_fusion
            - config.aggressive_fusion (will cause this function to be called more times)
        """
    @staticmethod
    def can_fuse_vertical(scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode, shared_data_score: int) -> bool:
        """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
    @staticmethod
    def can_fuse_horizontal(scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode, shared_data_score: int) -> bool:
        """Hook for heuristics to prevent horizontal (consumer/consumer) fusions"""
    @staticmethod
    def score_fusion(scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> Sortable:
        """
        Assign a score (higher comes first) to the fusion of node1 and node2.
        When different fusions conflict with each other, this is the way we
        decide what order to run them in.

        Our current score is based on:
        - The type of fusion (template/reduction/etc)
        - Estimate of the saved memory operations
        - Fusions closer together in original graph order
        """
