import torch

__all__: list[str]

def _input_requires_grad(*tensors: torch.Tensor) -> bool:
    """Returns True if any of the tensors requires grad"""
def _postprocess_flash_output(inpt_tensor: torch.Tensor, og_size: int) -> torch.Tensor:
    """Handles the unpad of the last dimension"""
def _calculate_scale(head_dim_size: int, scale: float | None) -> float:
    """
    For FlashAttention we pad the head dimension to be a multiple of 8 so we need to scale the output
    by the original head size and not the padded.
    """
def _validate_sdpa_input(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale=None): ...
