import torch
from torch._utils import _get_device_index as _get_device_index
from torch.autograd import Function as Function
from torch.nn.parallel import comm as comm

class Broadcast(Function):
    @staticmethod
    def forward(ctx, target_gpus, *inputs): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination, num_inputs, *grads): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class Gather(Function):
    @staticmethod
    def forward(ctx, target_device, dim, *inputs): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input): ...
    @staticmethod
    def backward(ctx, *grad_output): ...

_streams: list[torch.Stream | None] | None

def _get_stream(device: torch.device):
    """Get a background stream for copying between CPU and target device."""
