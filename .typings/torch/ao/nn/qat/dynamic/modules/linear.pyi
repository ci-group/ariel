import torch
from torch.ao.quantization.qconfig import QConfig

__all__ = ['Linear']

class Linear(torch.ao.nn.qat.Linear):
    """
    A linear module attached with FakeQuantize modules for weight,
    used for dynamic quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, qconfig: QConfig | None = None, device: int | str | torch.device | None = None, dtype: str | None = None) -> None: ...
