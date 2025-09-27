import torch
from _typeshed import Incomplete
from enum import Enum

TORCH_HALF_MIN: Incomplete
TORCH_HALF_MAX: Incomplete

class DQuantType(Enum):
    """
    Different quantization methods for auto_quantize API are identified here.

    auto_quantize API currently supports fp16 and bfp16 methods.
    """
    FP16 = ('fp16',)
    BFP16 = 'bfp16'
    def __str__(self) -> str: ...

def _fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor: ...
def _quantize_tensor(tensor, qtype): ...
def _quantize_tensor_list(tensor_list, qtype): ...
def _dequantize_tensor(tensor, qtype, quant_loss=None): ...
def _dequantize_tensor_list(tensor_list, qtype, quant_loss=None): ...
def auto_quantize(func, qtype, quant_loss=None):
    """
    Quantize the input tensors, choose the precision types, and pass other necessary arguments and then dequantizes the output.

    Currently it only supports:
        . FP16 and BFP16 quantization method supported for gloo and nccl backends
        . all_gather, all_to_all collective ops
    Note: BFP16 only supports 2D tensors.
    Args:
        func (Callable): A function representing collective operations.
        qtype (QuantType): Quantization method
        quant_loss (float, optional): This can be used to improve accuracy in the dequantization.
    Returns:
        (Callable): the same collective as func but enables automatic quantization/dequantization.
    """
