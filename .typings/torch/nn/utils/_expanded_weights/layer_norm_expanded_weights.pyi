import torch
from .expanded_weights_impl import ExpandedWeight as ExpandedWeight, implements_per_sample_grads as implements_per_sample_grads
from .expanded_weights_utils import forward_helper as forward_helper, set_grad_sample_if_exists as set_grad_sample_if_exists, standard_kwargs as standard_kwargs, sum_over_all_but_batch_and_last_n as sum_over_all_but_batch_and_last_n, unpack_expanded_weight_or_tensor as unpack_expanded_weight_or_tensor

class LayerNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs): ...
    @staticmethod
    def backward(ctx, grad_output): ...
