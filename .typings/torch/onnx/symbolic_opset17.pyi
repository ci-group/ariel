from collections.abc import Sequence
from torch import _C
from torch.onnx._internal import jit_utils

__all__ = ['layer_norm', 'stft', 'quantized_layer_norm']

def layer_norm(g: jit_utils.GraphContext, input: _C.Value, normalized_shape: Sequence[int], weight: _C.Value, bias: _C.Value, eps: float, cudnn_enable: bool): ...
def quantized_layer_norm(g: jit_utils.GraphContext, x, normalized_shape, weight, bias, eps, op_scale, op_zero_point): ...
def stft(g: jit_utils.GraphContext, input: _C.Value, n_fft: int, hop_length: int | None = None, win_length: int | None = None, window: _C.Value | None = None, normalized: bool = False, onesided: bool | None = True, return_complex: bool | None = False, align_to_window: bool | None = None) -> _C.Value:
    """Associates `torch.stft` with the `STFT` ONNX operator.
    Note that torch.stft calls _VF.stft, without centering or padding options.
    Hence, this function does not contain these two arguments.
    See torch.stft source code for more info.

    Args:
        g: Graph to write the ONNX representation into
        input: Input tensor for the transformation
        n_fft: FFT size
        hop_length: Size of the hop. Defaults to `floot(n_fft // 4)`
        win_length: Size of the analysis window. Defaults to `n_fft`
        window: Analysis window. Defaults to a window of all ones
        normalized: Whether to return a normalized STFT
        onesided: Whether to return only half (+1) of the results, given the
            symmetry of the STFT
        return_complex: Whether to return the complex value (Note: Must be
            `False` or `None`)

    Returns:
        op: Operator for torch.stft associated with STFT (ONNX)
    """
