import torch
from _typeshed import Incomplete

__all__ = ['ReferenceQuantizedModule']

class ReferenceQuantizedModule(torch.nn.Module):
    weight_qscheme: torch.qscheme
    weight_dtype: Incomplete
    is_decomposed: bool
    weight_axis_int: int
    weight_quant_min: int | None
    weight_quant_max: int | None
    def _init_weight_qparams(self, weight_qparams, device) -> None: ...
    def get_weight(self):
        """
        Fake quantize (quantize and dequantize) the weight with
        the quantization parameters for weight, this is used to
        simulate the numerics for the quantized weight in a quantized
        model
        """
    def get_quantized_weight(self): ...
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
