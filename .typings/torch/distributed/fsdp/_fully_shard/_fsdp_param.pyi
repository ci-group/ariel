import torch
import torch.nn as nn
from ._fsdp_api import CPUOffloadPolicy as CPUOffloadPolicy, MixedPrecisionPolicy as MixedPrecisionPolicy, OffloadPolicy as OffloadPolicy
from ._fsdp_common import FSDPMeshInfo as FSDPMeshInfo, HSDPMeshInfo as HSDPMeshInfo, _chunk_with_empty as _chunk_with_empty, _from_local_no_grad as _from_local_no_grad, _get_dim_chunked_size as _get_dim_chunked_size, _raise_assert_with_print as _raise_assert_with_print, _to_dtype_if_needed as _to_dtype_if_needed, compiled_autograd_enabled as compiled_autograd_enabled
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from torch._prims_common import make_contiguous_strides_for as make_contiguous_strides_for
from torch.distributed._functional_collectives import AsyncCollectiveTensor as AsyncCollectiveTensor
from torch.distributed.tensor import DTensor as DTensor, Replicate as Replicate, Shard as Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec as DTensorSpec, TensorMeta as TensorMeta
from torch.distributed.tensor.device_mesh import _mesh_resources as _mesh_resources
from torch.distributed.tensor.placement_types import Placement as Placement, _StridedShard as _StridedShard
from typing import Any, Callable

lib: Incomplete

def copy_(tensor, data) -> None: ...
def copy__functionalize(tensor, data) -> None: ...

class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``SHARDED_POST_FORWARD``: The unsharded parameter is resharded to a
      smaller world size. Since this data should not be used for computation,
      we do not register it to the module. Users should reshard the module
      before any in-place modifications. Both it and the sharded parameter
      contribute to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """
    SHARDED = ...
    SHARDED_POST_FORWARD = ...
    UNSHARDED = ...

@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``. We additionally save
    shared modules and shared parameter names to update them accordingly.
    """
    module: nn.Module
    param_name: str
    shared_modules: list[nn.Module] = field(default_factory=list)
    shared_param_names: list[str] = field(default_factory=list)

@dataclass
class ExtensionsData:
    all_gather_metadata: Any | None = ...
    all_gather_input_sizes: Sequence[torch.Size] = ...
    def clear(self) -> None: ...

class FSDPParam:
    """
    This class manages a parameter with FSDP or FSDP variants applied,
    implementing dim-0 per-parameter sharding.
    """
    orig_dtype: torch.dtype
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    _orig_size: torch.Size
    sharded_size: torch.Size
    contiguous_sharded_stride: tuple[int, ...]
    padded_sharded_param_size: torch.Size
    sharded_post_forward_size: torch.Size
    contiguous_sharded_post_forward_stride: tuple[int, ...]
    _sharded_param_data: torch.Tensor
    sharded_param: nn.Parameter
    _sharded_post_forward_param_data: torch.Tensor | None
    _sharded_post_forward_param: nn.Parameter | None
    _unsharded_param: nn.Parameter
    unsharded_accumulated_grad: torch.Tensor | None
    _sharding_spec: DTensorSpec
    _tp_spec: DTensorSpec
    all_gather_outputs: list[torch.Tensor]
    _extensions_data: ExtensionsData
    _unsharded_inner_tensors: list[torch.Tensor]
    _module_info: ParamModuleInfo
    mesh_info: Incomplete
    post_forward_mesh_info: Incomplete
    device: Incomplete
    mp_policy: Incomplete
    offload_to_cpu: bool
    pin_memory: Incomplete
    grad_offload_event: torch.Event | None
    _param_fqn: str | None
    _post_load_hook_handle: Incomplete
    def __init__(self, param: nn.Parameter, module_info: ParamModuleInfo, mesh_info: FSDPMeshInfo, post_forward_mesh_info: FSDPMeshInfo | None, device: torch.device, shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None, mp_policy: MixedPrecisionPolicy, offload_policy: OffloadPolicy) -> None: ...
    fsdp_placement: Incomplete
    is_dtensor: Incomplete
    _spmd_mesh: Incomplete
    _spmd_placements: tuple[Placement, ...]
    _contiguous_orig_stride: Incomplete
    sharded_state: Incomplete
    def _init_sharded_param(self, param: nn.Parameter, device: torch.device, shard_placement_fn: Callable | None): ...
    def _init_sharded_post_forward_param_metadata(self, param: torch.Tensor) -> None: ...
    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy): ...
    def _init_extensions(self) -> None: ...
    def init_all_gather_outputs(self, all_gather_input_numels: list[int], all_gather_input_dtypes: list[torch.dtype], world_size: int, device: torch.device, force_recreate: bool = False): ...
    def init_unsharded_param(self) -> None:
        """
        [Note: Invariants for torch.compile Traceable FSDP2]
        1. Under compile, we always re-populate the content of `self._unsharded_param`
           per AllGather using the slow path.
        2. Under compile, we always recreate `self.all_gather_outputs` per AllGather.
           This is to ensure the buffer creation is internal to the graph and
           avoid `self.all_gather_outputs` being captured as a graph input.
        3. Under compile, at the end of `free_unsharded_param()`, we always clean up
           `self.all_gather_outputs` and `self._unsharded_inner_tensors`,
           to avoid them being captured as graph output.

        With these invariants, only these tensors will be inputs to the graph:
        - Sharded parameters
        - Placeholders for the `self._unsharded_param` nn.Parameter
        """
    def _unflatten_all_gather_outputs(self) -> tuple[torch.Tensor, ...]: ...
    def to_sharded(self) -> None: ...
    def to_sharded_post_forward(self) -> None: ...
    def to_unsharded(self) -> None: ...
    def _setattr_on_modules(self, param: nn.Parameter) -> None: ...
    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
    def to_sharded_post_forward_dtensor(self, tensor: torch.Tensor) -> DTensor: ...
    def to_accumulated_grad_if_needed(self) -> None: ...
    def accumulate_unsharded_grad_if_needed(self) -> None: ...
    def alloc_all_gather_outputs(self) -> None: ...
    def free_unsharded_param(self) -> None: ...
    @property
    def all_gather_inputs(self) -> list[torch.Tensor]: ...
    @property
    def unsharded_param(self) -> nn.Parameter: ...
    @property
    def unsharded_grad_data(self) -> torch.Tensor: ...
    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor: ...
    def _get_grad_inner_tensor(self, grad: torch.Tensor) -> torch.Tensor: ...
    @property
    def _sharded_local_tensor(self) -> torch.Tensor: ...
    @property
    def shard_mesh(self): ...
    @property
    def shard_mesh_from_root(self): ...
    def _assert_in_states(self, *states: ShardedState) -> None: ...
    def reset_sharded_param(self) -> None: ...
    def __repr__(self) -> str: ...

def alloc_storage(tensor: torch.Tensor) -> None: ...
def free_storage(tensor: torch.Tensor) -> None: ...
def unsafe_setattr_param(module: nn.Module, param_name: str, param: nn.Parameter) -> None: ...
def set_requires_grad_if_needed(src_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> None: ...
