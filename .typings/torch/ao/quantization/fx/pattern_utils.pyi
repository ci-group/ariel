from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.utils import Pattern
from typing import Any

__all__ = ['get_default_fusion_patterns', 'get_default_quant_patterns', 'get_default_output_activation_post_process_map']

QuantizeHandler = Any

def get_default_fusion_patterns() -> dict[Pattern, QuantizeHandler]: ...
def get_default_quant_patterns() -> dict[Pattern, QuantizeHandler]: ...
def get_default_output_activation_post_process_map(is_training) -> dict[Pattern, ObserverBase]: ...
