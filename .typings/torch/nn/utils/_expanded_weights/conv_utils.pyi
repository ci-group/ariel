from .expanded_weights_utils import set_grad_sample_if_exists as set_grad_sample_if_exists, unpack_expanded_weight_or_tensor as unpack_expanded_weight_or_tensor

THRESHOLD: int

def conv_picker(func, conv1dOpt, conv2dOpt, conv3dOpt): ...
def conv_args_and_kwargs(kwarg_names, expanded_args_and_kwargs): ...
def conv_normalizer(input, weight, bias=None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1): ...
def conv_input_for_string_padding(func, padding_style, input, dilation, kernel_size): ...
def int_padding_for_string_padding(func, padding_style, dilation, kernel_size): ...
def conv_padding_for_same(dilation, kernel_size): ...
def conv_backward(func, ctx, grad_output): ...
def conv_unfold_weight_grad_sample(input, grad_output, weight_shape, kernel_size, stride, padding, dilation, groups, func): ...
def conv_group_weight_grad_sample(input, grad_output, weight_shape, stride, padding, dilation, batch_size, func): ...
def unfold3d(tensor, kernel_size, padding, stride, dilation):
    """
    Extract sliding local blocks from an batched input tensor.

    :class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
    This method implements the same action for 5D inputs
    Args:
        tensor: An input tensor of shape ``(B, C, D, H, W)``.
        kernel_size: the size of the sliding blocks
        padding: implicit zero padding to be added on both sides of input
        stride: the stride of the sliding blocks in the input spatial dimensions
        dilation: the spacing between the kernel points.
    Returns:
        A tensor of shape ``(B, C * np.prod(kernel_size), L)``, where L - output spatial dimensions.
        See :class:`torch.nn.Unfold` for more details
    Example:
        >>> # xdoctest: +SKIP
        >>> B, C, D, H, W = 3, 4, 5, 6, 7
        >>> tensor = torch.arange(1, B * C * D * H * W + 1.0).view(B, C, D, H, W)
        >>> unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
        torch.Size([3, 32, 120])
    """
