import torch
import torch.nn as nn
from ._fsdp_common import FSDPMeshInfo as FSDPMeshInfo, HSDPMeshInfo as HSDPMeshInfo, _is_composable_with_fsdp as _is_composable_with_fsdp
from ._fsdp_state import _get_module_fsdp_state as _get_module_fsdp_state
from _typeshed import Incomplete
from torch._logging import warning_once as warning_once
from torch.distributed.device_mesh import _get_device_handle as _get_device_handle
from torch.distributed.tensor import DTensor as DTensor, DeviceMesh as DeviceMesh, init_device_mesh as init_device_mesh
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass

logger: Incomplete

def _get_post_forward_mesh_info(reshard_after_forward: bool | int, mesh_info: FSDPMeshInfo) -> FSDPMeshInfo | None: ...
def _init_default_fully_shard_mesh() -> DeviceMesh:
    """Default to global CUDA mesh if possible else global CPU mesh."""
def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device: ...
def _ignore_module(module: nn.Module, ignored_params: set[nn.Parameter], ignore_decision: dict[nn.Module, bool]) -> bool:
    """
    Decide if it is safe to ignore a module for applying fully_shard.
    """
def _adjust_managed_modules(modules: list[nn.Module], ignored_params: set[nn.Parameter]) -> list[nn.Module]:
    """
    Adjust the given list of managed modules by removing those with all parameters ignored.
    """
def _get_managed_modules(root_modules: tuple[nn.Module, ...], ignored_params: set[nn.Parameter] | None = None) -> list[nn.Module]: ...
def _verify_managed_param(name: str, param: nn.Parameter) -> None:
    """
    Verify if the parameter is accepted by fully_shard. The only restriction now
    is that the parameter cannot be a scalar tensor (param.numel == 0) since we
    need at least one dim to shard.
    """
def _get_managed_states(modules: list[nn.Module], ignored_params: set[nn.Parameter] | None = None) -> tuple[list[nn.Parameter], list[torch.Tensor]]: ...
def _move_states_to_device(params: list[nn.Parameter], buffers: list[torch.Tensor], device: torch.device) -> None:
    """
    We have FSDP move states to device for simpler and faster initialization
    since FSDP almost always uses CUDA for training. We move parameters/buffers
    rather than modules since modules to support ignoring parameters/buffers in
    the future.
    """
