import torch.nn as nn
from torch.distributed._composable.contract import _get_registry as _get_registry
from torch.distributed.fsdp._common_utils import _FSDPState as _FSDPState, _get_module_fsdp_state as _get_module_fsdp_state

def _composable(module: nn.Module) -> bool:
    """
    Returns if ``module`` can compose with ``fully_shard``.
    """
def _get_fsdp_states_with_modules(module: nn.Module) -> tuple[list[_FSDPState], list[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the modules owning the states in the first list.

    For the wrapper code path, both returned lists are the same, each
    containing all ``FullyShardedDataParallel`` instances. For the composable
    code path, this returns a list of all composable state instances and a list
    of the corresponding fully sharded modules. See [Note: Fully Sharded
    Module].

    NOTE: The traversal does not proceed into any module annotated by an
    incompatible API (e.g. ``replicate``).
    """
def _get_fsdp_states(module: nn.Module) -> list[_FSDPState]:
    """See :func:`_get_fsdp_states_with_modules`."""
def _get_fsdp_handles(module: nn.Module) -> list:
    """
    Returns all ``FlatParamHandle`` s in the module tree rooted at ``module``
    following the rules in :func:`_get_fsdp_state`.
    """
