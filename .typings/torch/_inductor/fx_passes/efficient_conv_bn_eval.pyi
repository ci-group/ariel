import torch
import torch.nn as nn
from ..pattern_matcher import CallFunctionVarArgs as CallFunctionVarArgs, CallModuleVarArgs as CallModuleVarArgs, Match as Match, register_graph_pattern as register_graph_pattern
from .pre_grad import efficient_conv_bn_eval_pass as efficient_conv_bn_eval_pass
from torch._dynamo.utils import counters as counters
from torch.func import functional_call as functional_call

def efficient_conv_bn_eval(bn: nn.modules.batchnorm._BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor):
    '''
    Implementation based on https://arxiv.org/abs/2305.11624
    "Efficient ConvBN Blocks for Transfer Learning and Beyond"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
     reduces memory footprint and computation cost, at the cost of slightly
     reduced numerical stability.
    Args:
        bn (nn.modules.batchnorm._BatchNorm): a BatchNorm module.
        conv (nn.modules.conv._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    '''
def efficient_conv_bn_eval_decomposed(bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps, conv: torch._ops.OpOverload, conv_weight, conv_bias, x, conv_remainging_args):
    '''
    Implementation based on https://arxiv.org/abs/2305.11624
    "Efficient ConvBN Blocks for Transfer Learning and Beyond"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for **training** as well, but only if one sets `bn.training=False`. It
     reduces memory footprint and computation cost, at the cost of slightly
     reduced numerical stability.
    Args:
    '''
def efficient_conv_bn_eval_graph_transform_inlined(match: Match, *args, **kwargs): ...
def efficient_conv_bn_eval_graph_transform_decomposed(match: Match, *args, **kwargs): ...
def efficient_conv_bn_eval_graph_transform(match: Match, *args, **kwargs): ...
