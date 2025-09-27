import torch
from dataclasses import dataclass
from torch.autograd import Variable as Variable
from torch.distributed.utils import _free_storage as _free_storage
from typing import Any, no_type_check

@dataclass
class _AllreduceUpcastHookState:
    """
    State to manage DDP mixed precision in backward / gradient communication.

    This contains a weakref to the DDP module for access to reducer and process
    group, and a stream to run parameter and gradient upcasts.
    """
    ddp_weakref: Any
    upcast_stream: torch.Stream
    wait_for_stream_enqueued: bool = ...

@no_type_check
def _reducer_allreduce_and_upcast_hook(hook_state, bucket):
    """
    Perform allreduce in precision ``reduce_dtype``, upcast to prepare for optimizer.

    Performs allreduce in the reduced precision given by DDP's mixed precision
    reduce_dtype, and upcasts parameters and gradients to fp32 in preparation
    to run the optimizer.
    """
