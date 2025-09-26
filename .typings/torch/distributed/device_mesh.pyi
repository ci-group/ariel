import threading
import torch
import types
from _typeshed import Incomplete
from numpy.typing import ArrayLike
from torch._C._distributed_c10d import Backend as C10dBackend
from torch.distributed.distributed_c10d import ProcessGroup

__all__ = ['init_device_mesh', 'DeviceMesh']

class _DeviceMeshStub: ...

class _MeshEnv(threading.local):
    mesh_stack: list[DeviceMesh]
    child_to_root_mapping: dict[DeviceMesh, DeviceMesh]
    mesh_dim_group_options: dict[int, tuple[str, C10dBackend.Options | None]]
    root_to_flatten_mapping: dict[DeviceMesh, dict[str, DeviceMesh]]
    flatten_name_to_root_dims: dict[DeviceMesh, dict[str, tuple[int, ...]]]
    def __init__(self) -> None: ...
    def get_current_mesh(self) -> DeviceMesh: ...
    def create_sub_mesh(self, device_mesh: DeviceMesh, submesh_dim_names: tuple[str, ...], submesh_dims: list[tuple[int, ...]]) -> DeviceMesh: ...
    def create_flatten_mesh(self, device_mesh: DeviceMesh, mesh_dim_name: str | None = None) -> DeviceMesh: ...
    def get_root_mesh(self, device_mesh: DeviceMesh) -> DeviceMesh: ...
    def get_root_mesh_dim(self, device_mesh: DeviceMesh) -> int | None:
        """
            Returns the index of the mesh dim in the root mesh.
            The device_mesh passed in needs to be sliced out from the root mesh
            or submesh of the root mesh.
            """
    @staticmethod
    def num_devices_per_host(device_type: str) -> int: ...
    @staticmethod
    def num_hosts(device_type: str) -> int: ...
    def get_mesh_dim_by_name(self, device_mesh: DeviceMesh, mesh_dim_name: str) -> int: ...
    def _set_mesh_dim_group_options(self, dim: int, backend: str, pg_options: C10dBackend.Options | None = None) -> None: ...
    def _get_slice_mesh_dims(self, device_mesh, mesh_dim_names) -> list[tuple[int, ...]]:
        """
            Validate whether the mesh_dim_names is valid for slicing the given device_mesh.
            If valid, return dim indexes of the slice mesh in the device mesh.
            """
    def _get_all_submeshes(self, device_mesh: DeviceMesh, mesh_dim_name: str) -> list['DeviceMesh']:
        """
            Return all the submeshes of a given mesh dimension of the device mesh.
            """

class DeviceMesh:
    '''
        DeviceMesh represents a mesh of devices, where layout of devices could be
        represented as a n-d dimension array, and each value of the n-d dimensional
        array is the global id of the default process group ranks.

        DeviceMesh could be used to setup the N dimensional device connections across the cluster,
        and manage the ProcessGroups for N dimensional parallelisms. Communications could happen on
        each dimension of the DeviceMesh separately. DeviceMesh respects the device that user selects
        already (i.e. if user call `torch.cuda.set_device` before the DeviceMesh initialization),
        and will select/set the device for the current process if user does not set the device
        beforehand. Note that manual device selection should happen BEFORE the DeviceMesh initialization.

        DeviceMesh can also be used as a context manager when using together with DTensor APIs.

        .. note::
            DeviceMesh follows SPMD programming model, which means the same PyTorch Python program
            is running on all processes/ranks in the cluster. Therefore, users need to make sure the
            `mesh` array (which describes the layout of devices) should be identical across all ranks.
            Inconsistent `mesh` will lead to silent hang.

        Args:
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
            mesh (ndarray): A multi-dimensional array or an integer tensor describing the layout
                of devices, where the IDs are global IDs of the default process group.

        Returns:
            DeviceMesh: A :class:`DeviceMesh` object representing the device layout.

        The following program runs on each process/rank in an SPMD manner. In this example, we have 2
        hosts with 4 GPUs each.
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7).

        Example::

            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import DeviceMesh
            >>>
            >>> # Initialize device mesh as (2, 4) to represent the topology
            >>> # of cross-host(dim 0), and within-host (dim 1).
            >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
        '''
    device_type: str
    mesh: torch.Tensor
    mesh_dim_names: tuple[str, ...] | None
    _flatten_mesh_list: Incomplete
    _thread_id: Incomplete
    _coordinate_on_dim: list[int] | None
    def __init__(self, device_type: str, mesh: torch.Tensor | ArrayLike, *, mesh_dim_names: tuple[str, ...] | None = None, _init_backend: bool = True) -> None: ...
    def _setup_world_group_and_device(self): ...
    _dim_group_names: Incomplete
    def _init_process_groups(self) -> None: ...
    def __enter__(self) -> DeviceMesh: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: types.TracebackType | None) -> None: ...
    def __repr__(self) -> str: ...
    _hash: Incomplete
    def __hash__(self): ...
    def __eq__(self, other: object) -> bool: ...
    def __getitem__(self, mesh_dim_names: str | tuple[str, ...]) -> DeviceMesh:
        '''
            Slice the current DeviceMesh based on the mesh_dim_names given to create a submesh.
            The submesh created consists of the dimensions and the communicators indicated by
            ``mesh_dim_names``

            Args:
                mesh_dim_names (Union[str, Tuple[str]]): the name or the tuple of names of the
                mesh dimension of the DeviceMesh to create the submesh for.
            Returns:
                A :class:`DeviceMesh` object

            The following program runs on each process/rank in an SPMD manner in a world size of 8.
            In the first example:
                Calling mesh_2d["tp"] on rank 0, 1, 2, 3 returns a 1D submesh of DeviceMesh:([0, 1, 2, 3]).
                Calling mesh_2d["tp"] on rank 4, 5, 6, 7 returns a 1D submesh of  DeviceMesh:([4, 5, 6, 7]).
                Calling mesh_2d["dp"] on rank 0, 4 returns a 1D submesh of  DeviceMesh:([0, 4]).
                Calling mesh_2d["dp"] on rank 1, 5 returns a 1D submesh of  DeviceMesh:([1, 5]).
                Calling mesh_2d["dp"] on rank 2, 6 returns a 1D submesh of  DeviceMesh:([2, 6]).
                Calling mesh_2d["dp"] on rank 3, 7 returns a 1D submesh of  DeviceMesh:([3, 7]).

            In the second example:
                Calling mesh_3d["dp", "cp"] on rank 0, 1, 4, 5 returns a 2D submesh of DeviceMesh:([[0, 1], [4, 5]]).
                Calling mesh_3d["dp", "cp"] on rank 2, 3, 6, 7 returns a 2D submesh of DeviceMesh:([[2, 3], [6, 7]]).
                Calling mesh_3d["cp", "dp"] on rank 0, 1, 4, 5 returns a 2D submesh of DeviceMesh:([[0, 4], [1, 5]]).
                Calling mesh_3d["cp", "dp"] on rank 2, 3, 6, 7 returns a 2D submesh of DeviceMesh:([[2, 6], [3, 7]]).

            Example::

                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize a 2D device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh_2d = init_device_mesh(device_type="cuda", (2,4), mesh_dim_names=("dp", "tp"))
                >>> tp_mesh = mesh_2d["tp"]
                >>> dp_mesh = mesh_2d["dp"]
                >>>
                >>> # Initialize a 3D mesh.
                >>> mesh_3d = init_device_mesh(device_type="cuda", (2,2,2), mesh_dim_names=("dp", "pp", "cp"))
                >>> # The order of the mesh_dim_names provided deteremines the order of dimensions in the submesh.
                >>> dp_cp_mesh = mesh_3d["dp", "cp"]
                >>> cp_dp_mesh = mesh_3d["cp", "dp"]
            '''
    def get_group(self, mesh_dim: int | str | None = None) -> ProcessGroup:
        """
            Returns the single ProcessGroup specified by mesh_dim, or, if mesh_dim is not specified and the
            DeviceMesh is 1-dimensional, returns the only ProcessGroup in the mesh.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                A :class:`ProcessGroup` object.
            """
    def get_all_groups(self) -> list[ProcessGroup]:
        """
            Returns a list of ProcessGroups for all mesh dimensions.

            Returns:
                A list of :class:`ProcessGroup` object.
            """
    @staticmethod
    def from_group(group: ProcessGroup | list[ProcessGroup], device_type: str, mesh: torch.Tensor | ArrayLike | None = None, *, mesh_dim_names: tuple[str, ...] | None = None) -> DeviceMesh:
        '''
            Constructs a :class:`DeviceMesh` with ``device_type`` from an
            existing :class:`ProcessGroup` or a list of existing :class:`ProcessGroup`.

            The constructed device mesh has number of dimensions equal to the
            number of groups passed. For example, if a single process group is passed in,
            the resulted DeviceMesh is a 1D mesh. If a list of 2 process groups is passed in,
            the resulted DeviceMesh is a 2D mesh.

            If more than one group is passed, then the ``mesh`` and ``mesh_dim_names`` arguments
            are required. The order of the process groups passed in determines the topology of
            the mesh. For example, the first process group will be the 0th dimension of the DeviceMesh.
            The `mesh` tensor passed in must have the same number of dimensions as the number of process
            groups passed in, and the order of the dimensions in the `mesh` tensor must match the order
            in the process groups passed in.

            Args:
                group (ProcessGroup or list[ProcessGroup]): the existing ProcessGroup
                    or a list of existing ProcessGroups.
                device_type (str): The device type of the mesh. Currently supports: "cpu",
                    "cuda/cuda-like". Passing in a device type with a GPU index, such as "cuda:0",
                    is not allowed.
                mesh (torch.Tensor or ArrayLike, optional): A multi-dimensional array or an
                    integer tensor describing the layout of devices, where the IDs are global IDs
                    of the default process group. Default is None.
                mesh_dim_names (tuple[str], optional): A tuple of mesh dimension names to assign
                    to each dimension of the multi-dimensional array describing the layout of devices.
                    Its length must match the length of `mesh_shape`. Each string in `mesh_dim_names`
                    must be unique. Default is None.

            Returns:
                DeviceMesh: A :class:`DeviceMesh` object representing the device layout.
            '''
    def size(self, mesh_dim: int | None = None) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def get_rank(self) -> int:
        """
            Returns the current global rank.
            """
    def get_local_rank(self, mesh_dim: int | str | None = None) -> int:
        '''
            Returns the local rank of the given mesh_dim of the DeviceMesh.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                An integer denotes the local rank.

            The following program runs on each process/rank in an SPMD manner. In this example, we have 2
            hosts with 4 GPUs each.
            Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 0, 1, 2, 3 would return 0.
            Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 4, 5, 6, 7 would return 1.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 0, 4 would return 0.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 1, 5 would return 1.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 2, 6 would return 2.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 3, 7 would return 3.

            Example::

                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
            '''
    def get_coordinate(self) -> list[int] | None:
        """
            Return the relative indices of this rank relative to all
            dimensions of the mesh. If this rank is not part of the mesh, return None.
            """
    def _flatten(self, mesh_dim_name: str | None = None) -> DeviceMesh:
        '''
            Returns a 1D DeviceMesh by flattening the current DeviceMesh.

            If no mesh_dim_name is provided, the default is a string concatenating the mesh_dim_names of the
            given submesh with each mesh_dim_name separated by "_". For example, if we have a 3D mesh
            DeviceMesh([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], mesh_dim_names=("dp", "cp", "tp")), calling
            mesh_3d["dp", "cp"]._flatten() will create a 1D submesh DeviceMesh([0, 1, 2, 3], mesh_dim_names=("dp_cp",))
            on rank 0, 1, 2, 3 and a 1D submesh DeviceMesh([4, 5, 6, 7], mesh_dim_names=("dp_cp",)) on rank 4, 5, 6, 7.

            After the flattened dimension is created, to access the flattened dimension in mesh_3d, one can use the
            existing slicing method to obtain the flattened mesh through calling mesh_3d["dp_cp"].
            '''

def init_device_mesh(device_type: str, mesh_shape: tuple[int, ...], *, mesh_dim_names: tuple[str, ...] | None = None) -> DeviceMesh:
    '''
        Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.

        This creates a DeviceMesh with an n-dimensional array layout, where `n` is the length of `mesh_shape`.
        If `mesh_dim_names` is provided, each dimension is labeled as `mesh_dim_names[i]`.

        .. note::
            `init_device_mesh` follows SPMD programming model, meaning the same PyTorch Python program
            runs on all processes/ranks in the cluster. Ensure `mesh_shape` (the dimensions of the nD array
            describing device layout) is identical across all ranks. Inconsistent `mesh_shape` may lead to hanging.

        .. note::
            If no process group is found, init_device_mesh will initialize distributed process group/groups
            required for distributed communications behind the scene.

        Args:
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
                Passing in a device type with a GPU index, such as "cuda:0", is not allowed.
            mesh_shape (Tuple[int]): A tuple defining the dimensions of the multi-dimensional array
                describing the layout of devices.
            mesh_dim_names (Tuple[str], optional): A tuple of mesh dimension names to assign to each dimension
                of the multi-dimensional array describing the layout of devices. Its length must match the length
                of `mesh_shape`. Each string in `mesh_dim_names` must be unique.

        Returns:
            DeviceMesh: A :class:`DeviceMesh` object representing the device layout.

        Example::

            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import init_device_mesh
            >>>
            >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
            >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))

        '''
