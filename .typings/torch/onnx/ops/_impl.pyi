import torch
import typing
from _typeshed import Incomplete
from torch.onnx.ops import _dtype_mappings as _dtype_mappings
from typing import Callable

_T = typing.TypeVar('_T', bound=Callable)
ONNX_ATEN_DECOMP_TABLE: dict[torch._ops.OpOverload, Callable]
_ATTENTION_23_ALLOWED_INTERMEDIATE_PRECISIONS: Incomplete

def _onnx_op(op_type: str, opset_version: int) -> Callable[[_T], _T]:
    """Decorator to register an ONNX operator with a custom implementation."""
def rotary_embedding_23(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, position_ids: torch.Tensor | None = None, *, interleaved: bool = False, num_heads: int = 0, rotary_embedding_dim: int = 0) -> torch.Tensor:
    """RotaryEmbedding-23 https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html#rotaryembedding-23"""
def _get_scale_factor(scale: float | None, head_size: int) -> float:
    """Get the scale factor for attention computation."""
def _reshape_3d_to_4d(tensor: torch.Tensor, batch_size: int, num_heads: int) -> torch.Tensor:
    """Reshape 3D tensor to 4D for multi-head attention."""
def _get_qk_output_for_aten_spda(Q: torch.Tensor, K: torch.Tensor, current_q_num_heads: int, current_kv_num_heads: int, scale: float | None, qk_matmul_output_mode: int) -> torch.Tensor:
    """Get QK output tensor based on the specified mode."""
def _validate_gqa_configuration(current_q_num_heads: int, current_kv_num_heads: int) -> None:
    """Validate Group Query Attention configuration."""
def _compute_qk_output_for_mode_0(Q: torch.Tensor, K: torch.Tensor, current_q_num_heads: int, current_kv_num_heads: int, scale: float | None) -> torch.Tensor:
    """Helper function to compute QK output for qk_matmul_output_mode == 0."""
def attention_23(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor | None = None, past_key: torch.Tensor | None = None, past_value: torch.Tensor | None = None, *, is_causal: bool = False, kv_num_heads: int = 0, q_num_heads: int = 0, qk_matmul_output_mode: int = 0, scale: float | None = None, softcap: float = 0.0, softmax_precision: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention-23 https://onnx.ai/onnx/operators/onnx__Attention.html#attention-23"""
