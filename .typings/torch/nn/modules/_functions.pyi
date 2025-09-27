import torch
from _typeshed import Incomplete
from torch.autograd.function import Function as Function

class SyncBatchNorm(Function):
    process_group: Incomplete
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size): ...
    @staticmethod
    def backward(self, grad_output): ...

class CrossMapLRN2d(Function):
    @staticmethod
    def forward(ctx, input, size, alpha: float = 0.0001, beta: float = 0.75, k: int = 1): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class BackwardHookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args): ...
    @staticmethod
    def backward(ctx, *args): ...
