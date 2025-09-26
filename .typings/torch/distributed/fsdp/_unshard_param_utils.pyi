import contextlib
import torch.nn as nn
from ._flat_param import FlatParamHandle as FlatParamHandle
from collections.abc import Generator
from torch.distributed.fsdp._common_utils import HandleTrainingState as HandleTrainingState, TrainingState as TrainingState, _FSDPState as _FSDPState, _get_module_fsdp_state as _get_module_fsdp_state, _has_fsdp_params as _has_fsdp_params, _module_handle as _module_handle
from torch.distributed.fsdp._runtime_utils import _lazy_init as _lazy_init, _reset_flat_param_grad_info_if_needed as _reset_flat_param_grad_info_if_needed, _reshard as _reshard, _reshard_grads as _reshard_grads, _unshard as _unshard, _unshard_grads as _unshard_grads
from torch.distributed.utils import _p_assert as _p_assert

FLAT_PARAM: str

def _writeback_to_local_shard(handle: FlatParamHandle, writeback_grad: bool):
    """
    For the handle, writes back the this rank's shard of the unsharded
    flattened parameter to the sharded flattened parameter. If
    ``writeback_grad=True``, then writes back to the sharded gradient as
    well.

    Precondition: The handle's ``FlatParameter`` 's data points to the
    padded unsharded flattened parameter.
    """
def _deregister_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    De-registers the flattened parameter from the wrapped module, hiding it
    from ``nn.Module`` methods.

    We do not use ``del`` because we want ``FLAT_PARAM`` to always be an
    attribute but dynamically change whether it is visible to ``nn.Module``
    methods.
    """
def _register_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    Registers the flattened parameter to the wrapped module, making it
    visible to ``nn.Module`` methods.

    We do not use :meth:`nn.Module.register_parameter` because we want
    ``FLAT_PARAM`` to always be an attribute but dynamically change whether
    it is visible to ``nn.Module`` methods.
    """
@contextlib.contextmanager
def _unflatten_as_params(state: _FSDPState, module: nn.Module) -> Generator:
    """
    Assumes that the flattened parameter is unsharded. When in the context,
    de-registers the flattened parameter and unflattens the original
    parameters as ``nn.Parameter`` views into the flattened parameter.
    After the context, re-registers the flattened parameter and restores
    the original parameters as ``Tensor`` views into the flattened
    parameter.
    """
def _validate_unshard_params_args(state: _FSDPState, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None: ...
@contextlib.contextmanager
def _unshard_fsdp_state_params(module: nn.Module, state: _FSDPState, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    """
    This unshards the parameters for a single FSDP state ``state`` that
    corresponds to ``module``.
    """
@contextlib.contextmanager
def _unshard_params_for_summon(module: nn.Module, state: _FSDPState, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool): ...
@contextlib.contextmanager
def _unshard_params(module: nn.Module, recurse: bool, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    """
    This unshards FSDP-managed parameters for all modules with FSDP applied in
    the module tree rooted at ``module``.
    """
def _deregister_orig_params(state: _FSDPState, module: nn.Module) -> None:
    """
    Deregisters the original parameters; registers the ``FlatParameter``.
    """
def _register_orig_params(state: _FSDPState, module: nn.Module) -> None:
    """
    Deregisters the ``FlatParameter``; registers the original parameters.
    """
