from torch.distributed.device_mesh import _mesh_resources as _mesh_resources
from torch.distributed.tensor import DeviceMesh as DeviceMesh
from torch.distributed.tensor.placement_types import Placement as Placement

LayoutsType = Placement | tuple[Placement, ...]

def _deprecate_warnings(func_name: str, extra_msg: str) -> None:
    """
    Inject common validation logics for `_prepare_input` funcs via this decorator.

    Include verifying that input needs to be either a :class:`Tensor` or :class:`DTensor`
    and only 1D :class:`DeviceMesh` is passed in.
    """
def _validate_tp_mesh_dim(device_mesh: DeviceMesh) -> None:
    """
    Check whether TP mesh dimension is valid or not.

    Args:
        device_mesh (:class:`DeviceMesh`):
            The `device_mesh` where we perform
            Tensor Parallelism on.

    Return:
        `True` if the mesh dimension
        is valid, `False` otherwise.
    """
