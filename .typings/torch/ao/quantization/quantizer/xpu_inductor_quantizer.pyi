import functools
import torch
from torch.ao.quantization.quantizer.x86_inductor_quantizer import FilterFn, X86InductorQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.fx import Node

__all__ = ['XPUInductorQuantizer', 'get_default_xpu_inductor_quantization_config']

@functools.lru_cache
def get_default_xpu_inductor_quantization_config(): ...

class XPUInductorQuantizer(X86InductorQuantizer):
    """
    XPUInductorQuantizer is a class designed to facilitate
    quantization capability at Intel GPU backend. The class
    highly reuses the existing implementation of
    X86InductorQuantizer as both are intended to take advantage
    of the optimized kernels in oneDNN library.
    """
    def __init__(self) -> None: ...
    def _annotate_qat_conv2d_fusion_pattern(self, model: torch.fx.GraphModule, quantization_config: QuantizationConfig | None, filter_fn: FilterFn | None = None): ...
    def _annotate_maxpool2d(self, node: Node, quantization_config: QuantizationConfig | None) -> None:
        """
        Here we skip the annotate logic for maxpool at XPU backend
        as the quantized::max_pool2d is only implemented for CPU.
        """
    def _annotate_output_for_int8_in_int8_out_pattern(self, node: Node) -> None: ...
