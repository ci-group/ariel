import contextlib
import torch
from _typeshed import Incomplete
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec

__all__ = ['is_rng_supported_mesh', 'manual_seed', 'OffsetBasedRNGTracker']

def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    """Checks if the current device of ``device_mesh`` supports DTensor's random APIs.
    Currently DTensor Random APIs only supports cuda/cuda-like devices. We suggest
    users call this API to test the availability before using our random APIs.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh on which we check if the
            random ops APIs are supported.

    Returns:
        A bool value. True if ``device_mesh`` supports DTensor Random APIs; False otherwise.

    .. warning::
        Currently we only support correct RNG on cuda/cuda-like devices.
    """
def manual_seed(seed: int, device_mesh: DeviceMesh) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed. It is
            required that the ``device_mesh`` include the calling rank. This is
            to ensure that the SPMD region maintains a synchronous RNG state, which
            means no ranks should be initialized with values other than ``seed``.

    Returns:
        None

    .. warning::
        :func:`manual_seed` does not check the ``seed`` value correctness. Users must
        ensure on their own that the value passed in is the desired ``seed`` for ranks
        within ``device_mesh``.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        ``manual_seed`` will throw an error.
        Current implementation only supports a GPU device mesh.
    """

class _RNGStateTracker:
    """
    _RNGStateTracker stores Random Number Generator (RNG) state (a ByteTensor object)
    in a dict, mapping from a corresponding tag to each state tensor. It also provides
    a set of convenient utility methods to help access/modify the state tensors. The most
    important interface is _distribute_region which will be used when DTensor executes
    a random op (an operator that calls RNG).
    """
    _device: Incomplete
    _device_handle: Incomplete
    _states: dict[str, Tensor]
    _use_distribute_region: bool
    def __init__(self, device: torch.device) -> None: ...
    @property
    def rng_states(self) -> dict[str, Tensor]: ...
    @property
    def distribute_region_enabled(self) -> bool: ...
    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value) -> None: ...
    def rng_state_is_sync(self, name) -> bool: ...
    def get_seed(self, name: str) -> int: ...
    def set_seed(self, name: str, seed: int) -> None: ...
    def _distribute_region(self, spec: DTensorSpec): ...
    def _manual_seed(self, parallel_seed: int) -> None: ...

class OffsetBasedRNGTracker(_RNGStateTracker):
    """
    This subclass of ``_RNGStateTracker`` defines the default policy of how RNG states
    should be shared and synchronized among all ranks to respect the semantics of DTensor
    random operators.

    note: _RNGStateTracker only supports cuda/cuda-like device.
    """
    def __init__(self, device_mesh: DeviceMesh, run_state_sync: bool = True) -> None: ...
    def _manual_seed(self, parallel_seed: int) -> None: ...
    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec): ...
    def get_offset(self, name: str) -> int: ...
    def set_offset(self, name: str, offset: int) -> None: ...
    def _set_pre_op_offset(self, spec: DTensorSpec) -> None:
        """Set the starting RNG offset for current device's local shard before actual
        op execution. The pre_op_offset value should start from the current RNG offset
        and increment by the size of local shard until it reaches the size of the whole
        DTensor. For different ranks that hold the same DTensor shard, their pre_op_offset
        will be the same.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we prepare the offset for running random ops.

        Returns:
            None

        .. warning::
            Note that, current implementation does not consider DTensor's continguity.

        Example:
            take a DTensor of shape [8, 16] as an example. Assume that the DTensor
            is placed on a device mesh with placements ([Shard(1), Replicate(), Shard(0)]),
            and the mesh is:
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            ``spec.mesh.get_coordinate()`` provides the coordinate of the current rank
            in the mesh. For example, the coordinate of rank 5 is (1, 0, 1).

            Another concept to introduce besides rank coordinate is shard coordinate.
            Each rank holds a local shard of the DTensor. In the example, the DTensor
            is partitioned into 4 [4, 8] shards. The first shard has 2 replicas and
            rank 0 (coord (0, 0, 0)) and rank 2 (coord (0, 1, 0)) have 1 replica each.
            That being said, the local shard on rank 0 and rank 2 correspond to the same
            shard of the DTensor. To denote each DTensor shard, we use a shard coordinate
            (in the example, it will be a tuple (i, j) where shard (i, j) has the slice
            DTensor[4 * i : 4 * (i + 1), 8 * j : 8 * (j + 1)], 0 <= i < 2, 0 <= j < 2).

            Once we have rank coordinate and shard coordinate, we can calculate on each rank
            what shard of the DTensor the rank holds, with the help of dim_map. The dim_map
            of the above DTensor is [2, 0] so the shard coordinate of a rank with rank coord
            (x, y, z) is simply (z, x) by taking(rank_coord[dim_map[0]],rank_coord[dim_map[1]]).
            Following this calculation,
            rank 0 and rank 2 holds the shard of coord (0, 0);
            rank 1 and rank 3 holds the shard of coord (0, 1);
            rank 4 and rank 6 holds the shard of coord (1, 0);
            rank 5 and rank 7 holds the shard of coord (1, 1);

            The last value to calculate before obtaining the starting offset is the shard linear index.
            The starting offset for each rank will be its shard_linear_index * local_tensor_numel.
        """
    def _set_post_op_offset(self, spec: DTensorSpec, old_offset: int) -> None:
        """Sets the RNG to a synchronized state after running the local random op. Every
        rank should set its RNG offset to `old_offset + DTensor.numel()` where old_offset is
        the offset before calling `set_pre_op_offset` i.e. the offset before running DTensor
        random ops.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we post-process the offset for running random ops.

        Returns:
            None
        """
    def _calc_shard_linear_idx(self, shard_coord: list[int], shard_size: list[int]) -> int: ...
