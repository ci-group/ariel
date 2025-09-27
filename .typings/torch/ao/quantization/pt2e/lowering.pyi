import torch

__all__ = ['lower_pt2e_quantized_to_x86']

def lower_pt2e_quantized_to_x86(model: torch.fx.GraphModule, example_inputs: tuple[torch.Tensor, ...]) -> torch.fx.GraphModule:
    """Lower a PT2E-qantized model to x86 backend.

    Args:
    * `model` (torch.fx.GraphModule): a model quantized by PT2E quantization flow.
    * `example_inputs` (tuple[torch.Tensor, ...]): example inputs for the model.

    Return:
    A GraphModule lowered to x86 backend.
    """
