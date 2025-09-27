from enum import Enum

__all__ = ['DDPCommHookType', 'register_ddp_comm_hook']

class DDPCommHookType(Enum):
    """
    Enumerate ``ddp_comm_hooks`` and ``ddp_comm_hook_wrapper`` communucation hook types.

    DDPCommHookType enumerates the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
    as names and ``ddp_comm_hook_wrapper`` partials with hook specified. As an example,
    you can register allreduce hook by
    ``DDPCommHookType.ALLREDUCE.value(model=model, state=process_group)``.
    """
    ALLREDUCE = ...
    FP16_COMPRESS = ...
    BF16_COMPRESS = ...
    QUANTIZE_PER_TENSOR = ...
    QUANTIZE_PER_CHANNEL = ...
    POWER_SGD = ...
    POWER_SGD_RANK2 = ...
    BATCHED_POWER_SGD = ...
    BATCHED_POWER_SGD_RANK2 = ...
    NOOP = ...

def register_ddp_comm_hook(comm_hook_type: DDPCommHookType, model, state=None):
    """
    Register ``ddp_comm_hooks`` to DDP model.

    Registers the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
    to the DDP model. User can specify the type of hook as an enum
    ``DDPCommHookType`` type using ``comm_hook_type`` input. State input will
    be passed to the model.
    Uses Python comm hook implementations.

    Example::
        >>> # xdoctest: +SKIP
        >>> register_ddp_comm_hook(DDPCommHookType.FP16_COMPRESS, model, state)
    """
