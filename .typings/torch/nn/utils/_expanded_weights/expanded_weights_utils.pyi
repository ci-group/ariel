import torch
from .expanded_weights_impl import ExpandedWeight as ExpandedWeight

def is_batch_first(expanded_args_and_kwargs): ...
def standard_kwargs(kwarg_names, expanded_args):
    """Separate args and kwargs from `__torch_function__`s that standardize kwargs.

    Most `__torch_function__`s standardize the kwargs that they give, so this will separate
    the args and kwargs they pass. Functions that don't are linear and convND.
    """
def forward_helper(func, expanded_args, expanded_kwargs):
    """Compute the forward pass for a function that has expanded weight(s) passed to it.

    It will run the forward pass where all ExpandedWeights are their original
    weight. It runs checks on the given arguments and detaches the outputs.

    .. note:: First argument in :attr:`expanded_args` must be the input with the batch
    dimension as the first element of the shape

    .. note:: :attr:`func` must return a Tensor or tuple of Tensors

    Args:
        func: The function to be called
        expanded_args: Arguments to be passed to :attr:`func`. Will include arguments
          that need to be unpacked because they are ExpandedWeights
        expanded_kwargs: Keyword arguments to be passed to :attr:`func`.
          Similar to :attr:`expanded_args`.
    """
def _check_and_unexpand_args(func, expanded_args, expanded_kwargs): ...
def maybe_scale_by_batch_size(grad_sample, expanded_weight): ...
def set_grad_sample_if_exists(maybe_expanded_weight, per_sample_grad_fn) -> None: ...
def unpack_expanded_weight_or_tensor(maybe_expanded_weight, func=...): ...
def sum_over_all_but_batch_and_last_n(tensor: torch.Tensor, n_dims: int) -> torch.Tensor:
    """
    Calculate the sum over all dimensions, except the first (batch dimension), and excluding the last n_dims.

    This function will ignore the first dimension and it will
    not aggregate over the last n_dims dimensions.
    Args:
        tensor: An input tensor of shape ``(B, ..., X[n_dims-1])``.
        n_dims: Number of dimensions to keep.
    Example:
        >>> tensor = torch.ones(1, 2, 3, 4, 5)
        >>> sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape
        torch.Size([1, 4, 5])
    Returns:
        A tensor of shape ``(B, ..., X[n_dims-1])``
    """
