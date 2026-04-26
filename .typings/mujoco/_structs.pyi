import collections.abc
import mujoco._enums
import numpy
import numpy.typing
import typing
from typing import Callable, ClassVar, overload

class MjContact:
    H: numpy.typing.NDArray[numpy.float64]
    dim: int
    dist: float
    efc_address: int
    elem: numpy.typing.NDArray[numpy.int32]
    exclude: int
    flex: numpy.typing.NDArray[numpy.int32]
    frame: numpy.typing.NDArray[numpy.float64]
    friction: numpy.typing.NDArray[numpy.float64]
    geom: numpy.typing.NDArray[numpy.int32]
    geom1: int
    geom2: int
    includemargin: float
    mu: float
    pos: numpy.typing.NDArray[numpy.float64]
    solimp: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    solreffriction: numpy.typing.NDArray[numpy.float64]
    vert: numpy.typing.NDArray[numpy.int32]
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjContact) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjContact:
        """__copy__(self: mujoco._structs.MjContact) -> mujoco._structs.MjContact"""
    def __deepcopy__(self, arg0: dict) -> MjContact:
        """__deepcopy__(self: mujoco._structs.MjContact, arg0: dict) -> mujoco._structs.MjContact"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjData:
    bind: ClassVar[Callable] = ...
    M: numpy.typing.NDArray[numpy.float64]
    act: numpy.typing.NDArray[numpy.float64]
    act_dot: numpy.typing.NDArray[numpy.float64]
    actuator_force: numpy.typing.NDArray[numpy.float64]
    actuator_length: numpy.typing.NDArray[numpy.float64]
    actuator_moment: numpy.typing.NDArray[numpy.float64]
    actuator_velocity: numpy.typing.NDArray[numpy.float64]
    bvh_aabb_dyn: numpy.typing.NDArray[numpy.float64]
    bvh_active: numpy.typing.NDArray[numpy.uint8]
    cacc: numpy.typing.NDArray[numpy.float64]
    cam_xmat: numpy.typing.NDArray[numpy.float64]
    cam_xpos: numpy.typing.NDArray[numpy.float64]
    cdof: numpy.typing.NDArray[numpy.float64]
    cdof_dot: numpy.typing.NDArray[numpy.float64]
    cfrc_ext: numpy.typing.NDArray[numpy.float64]
    cfrc_int: numpy.typing.NDArray[numpy.float64]
    cinert: numpy.typing.NDArray[numpy.float64]
    crb: numpy.typing.NDArray[numpy.float64]
    ctrl: numpy.typing.NDArray[numpy.float64]
    cvel: numpy.typing.NDArray[numpy.float64]
    energy: numpy.typing.NDArray[numpy.float64]
    eq_active: numpy.typing.NDArray[numpy.uint8]
    flexedge_J: numpy.typing.NDArray[numpy.float64]
    flexedge_J_colind: numpy.typing.NDArray[numpy.int32]
    flexedge_J_rowadr: numpy.typing.NDArray[numpy.int32]
    flexedge_J_rownnz: numpy.typing.NDArray[numpy.int32]
    flexedge_length: numpy.typing.NDArray[numpy.float64]
    flexedge_velocity: numpy.typing.NDArray[numpy.float64]
    flexelem_aabb: numpy.typing.NDArray[numpy.float64]
    flexvert_xpos: numpy.typing.NDArray[numpy.float64]
    geom_xmat: numpy.typing.NDArray[numpy.float64]
    geom_xpos: numpy.typing.NDArray[numpy.float64]
    light_xdir: numpy.typing.NDArray[numpy.float64]
    light_xpos: numpy.typing.NDArray[numpy.float64]
    maxuse_arena: int
    maxuse_con: int
    maxuse_efc: int
    maxuse_stack: int
    maxuse_threadstack: numpy.typing.NDArray[numpy.uint64]
    mocap_pos: numpy.typing.NDArray[numpy.float64]
    mocap_quat: numpy.typing.NDArray[numpy.float64]
    moment_colind: numpy.typing.NDArray[numpy.int32]
    moment_rowadr: numpy.typing.NDArray[numpy.int32]
    moment_rownnz: numpy.typing.NDArray[numpy.int32]
    nA: int
    nJ: int
    narena: int
    nbuffer: int
    ncon: int
    ne: int
    nefc: int
    nf: int
    nidof: int
    nisland: int
    nl: int
    nplugin: int
    parena: int
    pbase: int
    plugin: numpy.typing.NDArray[numpy.int32]
    plugin_data: numpy.typing.NDArray[numpy.uint64]
    plugin_state: numpy.typing.NDArray[numpy.float64]
    pstack: int
    qDeriv: numpy.typing.NDArray[numpy.float64]
    qH: numpy.typing.NDArray[numpy.float64]
    qHDiagInv: numpy.typing.NDArray[numpy.float64]
    qLD: numpy.typing.NDArray[numpy.float64]
    qLDiagInv: numpy.typing.NDArray[numpy.float64]
    qLU: numpy.typing.NDArray[numpy.float64]
    qM: numpy.typing.NDArray[numpy.float64]
    qacc: numpy.typing.NDArray[numpy.float64]
    qacc_smooth: numpy.typing.NDArray[numpy.float64]
    qacc_warmstart: numpy.typing.NDArray[numpy.float64]
    qfrc_actuator: numpy.typing.NDArray[numpy.float64]
    qfrc_applied: numpy.typing.NDArray[numpy.float64]
    qfrc_bias: numpy.typing.NDArray[numpy.float64]
    qfrc_constraint: numpy.typing.NDArray[numpy.float64]
    qfrc_damper: numpy.typing.NDArray[numpy.float64]
    qfrc_fluid: numpy.typing.NDArray[numpy.float64]
    qfrc_gravcomp: numpy.typing.NDArray[numpy.float64]
    qfrc_inverse: numpy.typing.NDArray[numpy.float64]
    qfrc_passive: numpy.typing.NDArray[numpy.float64]
    qfrc_smooth: numpy.typing.NDArray[numpy.float64]
    qfrc_spring: numpy.typing.NDArray[numpy.float64]
    qpos: numpy.typing.NDArray[numpy.float64]
    qvel: numpy.typing.NDArray[numpy.float64]
    sensordata: numpy.typing.NDArray[numpy.float64]
    site_xmat: numpy.typing.NDArray[numpy.float64]
    site_xpos: numpy.typing.NDArray[numpy.float64]
    solver_fwdinv: numpy.typing.NDArray[numpy.float64]
    solver_niter: numpy.typing.NDArray[numpy.int32]
    solver_nnz: numpy.typing.NDArray[numpy.int32]
    subtree_angmom: numpy.typing.NDArray[numpy.float64]
    subtree_com: numpy.typing.NDArray[numpy.float64]
    subtree_linvel: numpy.typing.NDArray[numpy.float64]
    ten_J: numpy.typing.NDArray[numpy.float64]
    ten_J_colind: numpy.typing.NDArray[numpy.int32]
    ten_J_rowadr: numpy.typing.NDArray[numpy.int32]
    ten_J_rownnz: numpy.typing.NDArray[numpy.int32]
    ten_length: numpy.typing.NDArray[numpy.float64]
    ten_velocity: numpy.typing.NDArray[numpy.float64]
    ten_wrapadr: numpy.typing.NDArray[numpy.int32]
    ten_wrapnum: numpy.typing.NDArray[numpy.int32]
    threadpool: int
    time: float
    userdata: numpy.typing.NDArray[numpy.float64]
    wrap_obj: numpy.typing.NDArray[numpy.int32]
    wrap_xpos: numpy.typing.NDArray[numpy.float64]
    xanchor: numpy.typing.NDArray[numpy.float64]
    xaxis: numpy.typing.NDArray[numpy.float64]
    xfrc_applied: numpy.typing.NDArray[numpy.float64]
    ximat: numpy.typing.NDArray[numpy.float64]
    xipos: numpy.typing.NDArray[numpy.float64]
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    xquat: numpy.typing.NDArray[numpy.float64]
    def __init__(self, arg0: MjModel) -> None:
        """__init__(self: mujoco._structs.MjData, arg0: mujoco._structs.MjModel) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def actuator(self, *args, **kwargs):
        """actuator(*args, **kwargs)
        Overloaded function.

        1. actuator(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataActuatorViews

        2. actuator(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataActuatorViews
        """
    def bind_scalar(self, *args, **kwargs):
        """bind_scalar(*args, **kwargs)
        Overloaded function.

        1. bind_scalar(self: mujoco._structs.MjData, spec: mjsActuator_ = None) -> mujoco::python::MjDataActuatorViews

        2. bind_scalar(self: mujoco._structs.MjData, spec: mjsBody_ = None) -> mujoco::python::MjDataBodyViews

        3. bind_scalar(self: mujoco._structs.MjData, spec: mjsCamera_ = None) -> mujoco::python::MjDataCameraViews

        4. bind_scalar(self: mujoco._structs.MjData, spec: mjsGeom_ = None) -> mujoco::python::MjDataGeomViews

        5. bind_scalar(self: mujoco._structs.MjData, spec: mjsJoint_ = None) -> mujoco::python::MjDataJointViews

        6. bind_scalar(self: mujoco._structs.MjData, spec: mjsLight_ = None) -> mujoco::python::MjDataLightViews

        7. bind_scalar(self: mujoco._structs.MjData, spec: mjsSensor_ = None) -> mujoco::python::MjDataSensorViews

        8. bind_scalar(self: mujoco._structs.MjData, spec: mjsSite_ = None) -> mujoco::python::MjDataSiteViews

        9. bind_scalar(self: mujoco._structs.MjData, spec: mjsTendon_ = None) -> mujoco::python::MjDataTendonViews
        """
    def body(self, *args, **kwargs):
        """body(*args, **kwargs)
        Overloaded function.

        1. body(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataBodyViews

        2. body(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataBodyViews
        """
    def cam(self, *args, **kwargs):
        """cam(*args, **kwargs)
        Overloaded function.

        1. cam(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataCameraViews

        2. cam(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataCameraViews
        """
    def camera(self, *args, **kwargs):
        """camera(*args, **kwargs)
        Overloaded function.

        1. camera(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataCameraViews

        2. camera(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataCameraViews
        """
    def geom(self, *args, **kwargs):
        """geom(*args, **kwargs)
        Overloaded function.

        1. geom(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataGeomViews

        2. geom(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataGeomViews
        """
    def jnt(self, *args, **kwargs):
        """jnt(*args, **kwargs)
        Overloaded function.

        1. jnt(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataJointViews

        2. jnt(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataJointViews
        """
    def joint(self, *args, **kwargs):
        """joint(*args, **kwargs)
        Overloaded function.

        1. joint(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataJointViews

        2. joint(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataJointViews
        """
    def light(self, *args, **kwargs):
        """light(*args, **kwargs)
        Overloaded function.

        1. light(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataLightViews

        2. light(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataLightViews
        """
    def sensor(self, *args, **kwargs):
        """sensor(*args, **kwargs)
        Overloaded function.

        1. sensor(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataSensorViews

        2. sensor(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataSensorViews
        """
    def site(self, *args, **kwargs):
        """site(*args, **kwargs)
        Overloaded function.

        1. site(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataSiteViews

        2. site(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataSiteViews
        """
    def ten(self, *args, **kwargs):
        """ten(*args, **kwargs)
        Overloaded function.

        1. ten(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataTendonViews

        2. ten(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataTendonViews
        """
    def tendon(self, *args, **kwargs):
        """tendon(*args, **kwargs)
        Overloaded function.

        1. tendon(self: mujoco._structs.MjData, arg0: typing.SupportsInt) -> mujoco::python::MjDataTendonViews

        2. tendon(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataTendonViews
        """
    def __copy__(self) -> MjData:
        """__copy__(self: mujoco._structs.MjData) -> mujoco._structs.MjData"""
    def __deepcopy__(self, arg0: dict) -> MjData:
        """__deepcopy__(self: mujoco._structs.MjData, arg0: dict) -> mujoco._structs.MjData"""
    @property
    def _address(self) -> int:
        """(arg0: mujoco._structs.MjData) -> int"""
    @property
    def contact(self) -> _MjContactList:
        """(arg0: mujoco._structs.MjData) -> mujoco._structs._MjContactList"""
    @property
    def dof_island(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_AR(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_AR_colind(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_AR_rowadr(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_AR_rownnz(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_D(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_J(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_J_colind(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_J_rowadr(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_J_rownnz(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_J_rowsuper(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_KBIP(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_R(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_aref(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_b(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_diagApprox(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_force(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_frictionloss(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_id(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_island(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_margin(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_pos(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_state(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_type(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def efc_vel(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iLD(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iLDiagInv(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iM(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iM_colind(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iM_rowadr(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iM_rownnz(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iacc(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iacc_smooth(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iefc_D(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iefc_J(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iefc_J_colind(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iefc_J_rowadr(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iefc_J_rownnz(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iefc_J_rowsuper(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iefc_R(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iefc_aref(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iefc_force(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iefc_frictionloss(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def iefc_id(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iefc_state(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def iefc_type(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def ifrc_constraint(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def ifrc_smooth(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def island_dofadr(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def island_idofadr(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def island_iefcadr(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def island_ne(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def island_nefc(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def island_nf(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def island_nv(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def map_dof2idof(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def map_efc2iefc(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def map_idof2dof(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def map_iefc2efc(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def model(self) -> MjModel:
        """(arg0: mujoco._structs.MjData) -> mujoco._structs.MjModel"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._structs.MjData) -> int"""
    @property
    def solver(self) -> _MjSolverStatList:
        """(arg0: mujoco._structs.MjData) -> mujoco._structs._MjSolverStatList"""
    @property
    def tendon_efcadr(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs.MjData) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def timer(self) -> _MjTimerStatList:
        """(arg0: mujoco._structs.MjData) -> mujoco._structs._MjTimerStatList"""
    @property
    def warning(self) -> _MjWarningStatList:
        """(arg0: mujoco._structs.MjData) -> mujoco._structs._MjWarningStatList"""

class MjLROpt:
    accel: float
    interval: float
    inttotal: float
    maxforce: float
    mode: int
    timeconst: float
    timestep: float
    tolrange: float
    useexisting: int
    uselimit: int
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjLROpt) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjLROpt:
        """__copy__(self: mujoco._structs.MjLROpt) -> mujoco._structs.MjLROpt"""
    def __deepcopy__(self, arg0: dict) -> MjLROpt:
        """__deepcopy__(self: mujoco._structs.MjLROpt, arg0: dict) -> mujoco._structs.MjLROpt"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjModel:
    _size_fields: ClassVar[tuple] = ...  # read-only
    bind: ClassVar[Callable] = ...
    B_colind: numpy.typing.NDArray[numpy.int32]
    B_rowadr: numpy.typing.NDArray[numpy.int32]
    B_rownnz: numpy.typing.NDArray[numpy.int32]
    D_colind: numpy.typing.NDArray[numpy.int32]
    D_diag: numpy.typing.NDArray[numpy.int32]
    D_rowadr: numpy.typing.NDArray[numpy.int32]
    D_rownnz: numpy.typing.NDArray[numpy.int32]
    M_colind: numpy.typing.NDArray[numpy.int32]
    M_rowadr: numpy.typing.NDArray[numpy.int32]
    M_rownnz: numpy.typing.NDArray[numpy.int32]
    actuator_acc0: numpy.typing.NDArray[numpy.float64]
    actuator_actadr: numpy.typing.NDArray[numpy.int32]
    actuator_actearly: numpy.typing.NDArray[numpy.uint8]
    actuator_actlimited: numpy.typing.NDArray[numpy.uint8]
    actuator_actnum: numpy.typing.NDArray[numpy.int32]
    actuator_actrange: numpy.typing.NDArray[numpy.float64]
    actuator_biasprm: numpy.typing.NDArray[numpy.float64]
    actuator_biastype: numpy.typing.NDArray[numpy.int32]
    actuator_cranklength: numpy.typing.NDArray[numpy.float64]
    actuator_ctrllimited: numpy.typing.NDArray[numpy.uint8]
    actuator_ctrlrange: numpy.typing.NDArray[numpy.float64]
    actuator_dynprm: numpy.typing.NDArray[numpy.float64]
    actuator_dyntype: numpy.typing.NDArray[numpy.int32]
    actuator_forcelimited: numpy.typing.NDArray[numpy.uint8]
    actuator_forcerange: numpy.typing.NDArray[numpy.float64]
    actuator_gainprm: numpy.typing.NDArray[numpy.float64]
    actuator_gaintype: numpy.typing.NDArray[numpy.int32]
    actuator_gear: numpy.typing.NDArray[numpy.float64]
    actuator_group: numpy.typing.NDArray[numpy.int32]
    actuator_length0: numpy.typing.NDArray[numpy.float64]
    actuator_lengthrange: numpy.typing.NDArray[numpy.float64]
    actuator_plugin: numpy.typing.NDArray[numpy.int32]
    actuator_trnid: numpy.typing.NDArray[numpy.int32]
    actuator_trntype: numpy.typing.NDArray[numpy.int32]
    actuator_user: numpy.typing.NDArray[numpy.float64]
    body_bvhadr: numpy.typing.NDArray[numpy.int32]
    body_bvhnum: numpy.typing.NDArray[numpy.int32]
    body_conaffinity: numpy.typing.NDArray[numpy.int32]
    body_contype: numpy.typing.NDArray[numpy.int32]
    body_dofadr: numpy.typing.NDArray[numpy.int32]
    body_dofnum: numpy.typing.NDArray[numpy.int32]
    body_geomadr: numpy.typing.NDArray[numpy.int32]
    body_geomnum: numpy.typing.NDArray[numpy.int32]
    body_gravcomp: numpy.typing.NDArray[numpy.float64]
    body_inertia: numpy.typing.NDArray[numpy.float64]
    body_invweight0: numpy.typing.NDArray[numpy.float64]
    body_ipos: numpy.typing.NDArray[numpy.float64]
    body_iquat: numpy.typing.NDArray[numpy.float64]
    body_jntadr: numpy.typing.NDArray[numpy.int32]
    body_jntnum: numpy.typing.NDArray[numpy.int32]
    body_margin: numpy.typing.NDArray[numpy.float64]
    body_mass: numpy.typing.NDArray[numpy.float64]
    body_mocapid: numpy.typing.NDArray[numpy.int32]
    body_parentid: numpy.typing.NDArray[numpy.int32]
    body_plugin: numpy.typing.NDArray[numpy.int32]
    body_pos: numpy.typing.NDArray[numpy.float64]
    body_quat: numpy.typing.NDArray[numpy.float64]
    body_rootid: numpy.typing.NDArray[numpy.int32]
    body_sameframe: numpy.typing.NDArray[numpy.uint8]
    body_simple: numpy.typing.NDArray[numpy.uint8]
    body_subtreemass: numpy.typing.NDArray[numpy.float64]
    body_treeid: numpy.typing.NDArray[numpy.int32]
    body_user: numpy.typing.NDArray[numpy.float64]
    body_weldid: numpy.typing.NDArray[numpy.int32]
    bvh_aabb: numpy.typing.NDArray[numpy.float64]
    bvh_child: numpy.typing.NDArray[numpy.int32]
    bvh_depth: numpy.typing.NDArray[numpy.int32]
    bvh_nodeid: numpy.typing.NDArray[numpy.int32]
    cam_bodyid: numpy.typing.NDArray[numpy.int32]
    cam_fovy: numpy.typing.NDArray[numpy.float64]
    cam_intrinsic: numpy.typing.NDArray[numpy.float32]
    cam_ipd: numpy.typing.NDArray[numpy.float64]
    cam_mat0: numpy.typing.NDArray[numpy.float64]
    cam_mode: numpy.typing.NDArray[numpy.int32]
    cam_orthographic: numpy.typing.NDArray[numpy.int32]
    cam_pos: numpy.typing.NDArray[numpy.float64]
    cam_pos0: numpy.typing.NDArray[numpy.float64]
    cam_poscom0: numpy.typing.NDArray[numpy.float64]
    cam_quat: numpy.typing.NDArray[numpy.float64]
    cam_resolution: numpy.typing.NDArray[numpy.int32]
    cam_sensorsize: numpy.typing.NDArray[numpy.float32]
    cam_targetbodyid: numpy.typing.NDArray[numpy.int32]
    cam_user: numpy.typing.NDArray[numpy.float64]
    dof_M0: numpy.typing.NDArray[numpy.float64]
    dof_Madr: numpy.typing.NDArray[numpy.int32]
    dof_armature: numpy.typing.NDArray[numpy.float64]
    dof_bodyid: numpy.typing.NDArray[numpy.int32]
    dof_damping: numpy.typing.NDArray[numpy.float64]
    dof_frictionloss: numpy.typing.NDArray[numpy.float64]
    dof_invweight0: numpy.typing.NDArray[numpy.float64]
    dof_jntid: numpy.typing.NDArray[numpy.int32]
    dof_parentid: numpy.typing.NDArray[numpy.int32]
    dof_simplenum: numpy.typing.NDArray[numpy.int32]
    dof_solimp: numpy.typing.NDArray[numpy.float64]
    dof_solref: numpy.typing.NDArray[numpy.float64]
    dof_treeid: numpy.typing.NDArray[numpy.int32]
    eq_active0: numpy.typing.NDArray[numpy.uint8]
    eq_data: numpy.typing.NDArray[numpy.float64]
    eq_obj1id: numpy.typing.NDArray[numpy.int32]
    eq_obj2id: numpy.typing.NDArray[numpy.int32]
    eq_objtype: numpy.typing.NDArray[numpy.int32]
    eq_solimp: numpy.typing.NDArray[numpy.float64]
    eq_solref: numpy.typing.NDArray[numpy.float64]
    eq_type: numpy.typing.NDArray[numpy.int32]
    exclude_signature: numpy.typing.NDArray[numpy.int32]
    flex_activelayers: numpy.typing.NDArray[numpy.int32]
    flex_bending: numpy.typing.NDArray[numpy.float64]
    flex_bvhadr: numpy.typing.NDArray[numpy.int32]
    flex_bvhnum: numpy.typing.NDArray[numpy.int32]
    flex_centered: numpy.typing.NDArray[numpy.uint8]
    flex_conaffinity: numpy.typing.NDArray[numpy.int32]
    flex_condim: numpy.typing.NDArray[numpy.int32]
    flex_contype: numpy.typing.NDArray[numpy.int32]
    flex_damping: numpy.typing.NDArray[numpy.float64]
    flex_dim: numpy.typing.NDArray[numpy.int32]
    flex_edge: numpy.typing.NDArray[numpy.int32]
    flex_edgeadr: numpy.typing.NDArray[numpy.int32]
    flex_edgedamping: numpy.typing.NDArray[numpy.float64]
    flex_edgeequality: numpy.typing.NDArray[numpy.uint8]
    flex_edgeflap: numpy.typing.NDArray[numpy.int32]
    flex_edgenum: numpy.typing.NDArray[numpy.int32]
    flex_edgestiffness: numpy.typing.NDArray[numpy.float64]
    flex_elem: numpy.typing.NDArray[numpy.int32]
    flex_elemadr: numpy.typing.NDArray[numpy.int32]
    flex_elemdataadr: numpy.typing.NDArray[numpy.int32]
    flex_elemedge: numpy.typing.NDArray[numpy.int32]
    flex_elemedgeadr: numpy.typing.NDArray[numpy.int32]
    flex_elemlayer: numpy.typing.NDArray[numpy.int32]
    flex_elemnum: numpy.typing.NDArray[numpy.int32]
    flex_elemtexcoord: numpy.typing.NDArray[numpy.int32]
    flex_evpair: numpy.typing.NDArray[numpy.int32]
    flex_evpairadr: numpy.typing.NDArray[numpy.int32]
    flex_evpairnum: numpy.typing.NDArray[numpy.int32]
    flex_flatskin: numpy.typing.NDArray[numpy.uint8]
    flex_friction: numpy.typing.NDArray[numpy.float64]
    flex_gap: numpy.typing.NDArray[numpy.float64]
    flex_group: numpy.typing.NDArray[numpy.int32]
    flex_internal: numpy.typing.NDArray[numpy.uint8]
    flex_interp: numpy.typing.NDArray[numpy.int32]
    flex_margin: numpy.typing.NDArray[numpy.float64]
    flex_matid: numpy.typing.NDArray[numpy.int32]
    flex_node: numpy.typing.NDArray[numpy.float64]
    flex_node0: numpy.typing.NDArray[numpy.float64]
    flex_nodeadr: numpy.typing.NDArray[numpy.int32]
    flex_nodebodyid: numpy.typing.NDArray[numpy.int32]
    flex_nodenum: numpy.typing.NDArray[numpy.int32]
    flex_passive: numpy.typing.NDArray[numpy.int32]
    flex_priority: numpy.typing.NDArray[numpy.int32]
    flex_radius: numpy.typing.NDArray[numpy.float64]
    flex_rgba: numpy.typing.NDArray[numpy.float32]
    flex_rigid: numpy.typing.NDArray[numpy.uint8]
    flex_selfcollide: numpy.typing.NDArray[numpy.int32]
    flex_shell: numpy.typing.NDArray[numpy.int32]
    flex_shelldataadr: numpy.typing.NDArray[numpy.int32]
    flex_shellnum: numpy.typing.NDArray[numpy.int32]
    flex_solimp: numpy.typing.NDArray[numpy.float64]
    flex_solmix: numpy.typing.NDArray[numpy.float64]
    flex_solref: numpy.typing.NDArray[numpy.float64]
    flex_stiffness: numpy.typing.NDArray[numpy.float64]
    flex_texcoord: numpy.typing.NDArray[numpy.float32]
    flex_texcoordadr: numpy.typing.NDArray[numpy.int32]
    flex_vert: numpy.typing.NDArray[numpy.float64]
    flex_vert0: numpy.typing.NDArray[numpy.float64]
    flex_vertadr: numpy.typing.NDArray[numpy.int32]
    flex_vertbodyid: numpy.typing.NDArray[numpy.int32]
    flex_vertnum: numpy.typing.NDArray[numpy.int32]
    flexedge_invweight0: numpy.typing.NDArray[numpy.float64]
    flexedge_length0: numpy.typing.NDArray[numpy.float64]
    flexedge_rigid: numpy.typing.NDArray[numpy.uint8]
    geom_aabb: numpy.typing.NDArray[numpy.float64]
    geom_bodyid: numpy.typing.NDArray[numpy.int32]
    geom_conaffinity: numpy.typing.NDArray[numpy.int32]
    geom_condim: numpy.typing.NDArray[numpy.int32]
    geom_contype: numpy.typing.NDArray[numpy.int32]
    geom_dataid: numpy.typing.NDArray[numpy.int32]
    geom_fluid: numpy.typing.NDArray[numpy.float64]
    geom_friction: numpy.typing.NDArray[numpy.float64]
    geom_gap: numpy.typing.NDArray[numpy.float64]
    geom_group: numpy.typing.NDArray[numpy.int32]
    geom_margin: numpy.typing.NDArray[numpy.float64]
    geom_matid: numpy.typing.NDArray[numpy.int32]
    geom_plugin: numpy.typing.NDArray[numpy.int32]
    geom_pos: numpy.typing.NDArray[numpy.float64]
    geom_priority: numpy.typing.NDArray[numpy.int32]
    geom_quat: numpy.typing.NDArray[numpy.float64]
    geom_rbound: numpy.typing.NDArray[numpy.float64]
    geom_rgba: numpy.typing.NDArray[numpy.float32]
    geom_sameframe: numpy.typing.NDArray[numpy.uint8]
    geom_size: numpy.typing.NDArray[numpy.float64]
    geom_solimp: numpy.typing.NDArray[numpy.float64]
    geom_solmix: numpy.typing.NDArray[numpy.float64]
    geom_solref: numpy.typing.NDArray[numpy.float64]
    geom_type: numpy.typing.NDArray[numpy.int32]
    geom_user: numpy.typing.NDArray[numpy.float64]
    hfield_adr: numpy.typing.NDArray[numpy.int32]
    hfield_data: numpy.typing.NDArray[numpy.float32]
    hfield_ncol: numpy.typing.NDArray[numpy.int32]
    hfield_nrow: numpy.typing.NDArray[numpy.int32]
    hfield_pathadr: numpy.typing.NDArray[numpy.int32]
    hfield_size: numpy.typing.NDArray[numpy.float64]
    jnt_actfrclimited: numpy.typing.NDArray[numpy.uint8]
    jnt_actfrcrange: numpy.typing.NDArray[numpy.float64]
    jnt_actgravcomp: numpy.typing.NDArray[numpy.uint8]
    jnt_axis: numpy.typing.NDArray[numpy.float64]
    jnt_bodyid: numpy.typing.NDArray[numpy.int32]
    jnt_dofadr: numpy.typing.NDArray[numpy.int32]
    jnt_group: numpy.typing.NDArray[numpy.int32]
    jnt_limited: numpy.typing.NDArray[numpy.uint8]
    jnt_margin: numpy.typing.NDArray[numpy.float64]
    jnt_pos: numpy.typing.NDArray[numpy.float64]
    jnt_qposadr: numpy.typing.NDArray[numpy.int32]
    jnt_range: numpy.typing.NDArray[numpy.float64]
    jnt_solimp: numpy.typing.NDArray[numpy.float64]
    jnt_solref: numpy.typing.NDArray[numpy.float64]
    jnt_stiffness: numpy.typing.NDArray[numpy.float64]
    jnt_type: numpy.typing.NDArray[numpy.int32]
    jnt_user: numpy.typing.NDArray[numpy.float64]
    key_act: numpy.typing.NDArray[numpy.float64]
    key_ctrl: numpy.typing.NDArray[numpy.float64]
    key_mpos: numpy.typing.NDArray[numpy.float64]
    key_mquat: numpy.typing.NDArray[numpy.float64]
    key_qpos: numpy.typing.NDArray[numpy.float64]
    key_qvel: numpy.typing.NDArray[numpy.float64]
    key_time: numpy.typing.NDArray[numpy.float64]
    light_active: numpy.typing.NDArray[numpy.uint8]
    light_ambient: numpy.typing.NDArray[numpy.float32]
    light_attenuation: numpy.typing.NDArray[numpy.float32]
    light_bodyid: numpy.typing.NDArray[numpy.int32]
    light_bulbradius: numpy.typing.NDArray[numpy.float32]
    light_castshadow: numpy.typing.NDArray[numpy.uint8]
    light_cutoff: numpy.typing.NDArray[numpy.float32]
    light_diffuse: numpy.typing.NDArray[numpy.float32]
    light_dir: numpy.typing.NDArray[numpy.float64]
    light_dir0: numpy.typing.NDArray[numpy.float64]
    light_exponent: numpy.typing.NDArray[numpy.float32]
    light_intensity: numpy.typing.NDArray[numpy.float32]
    light_mode: numpy.typing.NDArray[numpy.int32]
    light_pos: numpy.typing.NDArray[numpy.float64]
    light_pos0: numpy.typing.NDArray[numpy.float64]
    light_poscom0: numpy.typing.NDArray[numpy.float64]
    light_range: numpy.typing.NDArray[numpy.float32]
    light_specular: numpy.typing.NDArray[numpy.float32]
    light_targetbodyid: numpy.typing.NDArray[numpy.int32]
    light_texid: numpy.typing.NDArray[numpy.int32]
    light_type: numpy.typing.NDArray[numpy.int32]
    mapD2M: numpy.typing.NDArray[numpy.int32]
    mapM2D: numpy.typing.NDArray[numpy.int32]
    mapM2M: numpy.typing.NDArray[numpy.int32]
    mat_emission: numpy.typing.NDArray[numpy.float32]
    mat_metallic: numpy.typing.NDArray[numpy.float32]
    mat_reflectance: numpy.typing.NDArray[numpy.float32]
    mat_rgba: numpy.typing.NDArray[numpy.float32]
    mat_roughness: numpy.typing.NDArray[numpy.float32]
    mat_shininess: numpy.typing.NDArray[numpy.float32]
    mat_specular: numpy.typing.NDArray[numpy.float32]
    mat_texid: numpy.typing.NDArray[numpy.int32]
    mat_texrepeat: numpy.typing.NDArray[numpy.float32]
    mat_texuniform: numpy.typing.NDArray[numpy.uint8]
    mesh_bvhadr: numpy.typing.NDArray[numpy.int32]
    mesh_bvhnum: numpy.typing.NDArray[numpy.int32]
    mesh_face: numpy.typing.NDArray[numpy.int32]
    mesh_faceadr: numpy.typing.NDArray[numpy.int32]
    mesh_facenormal: numpy.typing.NDArray[numpy.int32]
    mesh_facenum: numpy.typing.NDArray[numpy.int32]
    mesh_facetexcoord: numpy.typing.NDArray[numpy.int32]
    mesh_graph: numpy.typing.NDArray[numpy.int32]
    mesh_graphadr: numpy.typing.NDArray[numpy.int32]
    mesh_normal: numpy.typing.NDArray[numpy.float32]
    mesh_normaladr: numpy.typing.NDArray[numpy.int32]
    mesh_normalnum: numpy.typing.NDArray[numpy.int32]
    mesh_octadr: numpy.typing.NDArray[numpy.int32]
    mesh_octnum: numpy.typing.NDArray[numpy.int32]
    mesh_pathadr: numpy.typing.NDArray[numpy.int32]
    mesh_polyadr: numpy.typing.NDArray[numpy.int32]
    mesh_polymap: numpy.typing.NDArray[numpy.int32]
    mesh_polymapadr: numpy.typing.NDArray[numpy.int32]
    mesh_polymapnum: numpy.typing.NDArray[numpy.int32]
    mesh_polynormal: numpy.typing.NDArray[numpy.float64]
    mesh_polynum: numpy.typing.NDArray[numpy.int32]
    mesh_polyvert: numpy.typing.NDArray[numpy.int32]
    mesh_polyvertadr: numpy.typing.NDArray[numpy.int32]
    mesh_polyvertnum: numpy.typing.NDArray[numpy.int32]
    mesh_pos: numpy.typing.NDArray[numpy.float64]
    mesh_quat: numpy.typing.NDArray[numpy.float64]
    mesh_scale: numpy.typing.NDArray[numpy.float64]
    mesh_texcoord: numpy.typing.NDArray[numpy.float32]
    mesh_texcoordadr: numpy.typing.NDArray[numpy.int32]
    mesh_texcoordnum: numpy.typing.NDArray[numpy.int32]
    mesh_vert: numpy.typing.NDArray[numpy.float32]
    mesh_vertadr: numpy.typing.NDArray[numpy.int32]
    mesh_vertnum: numpy.typing.NDArray[numpy.int32]
    name_actuatoradr: numpy.typing.NDArray[numpy.int32]
    name_bodyadr: numpy.typing.NDArray[numpy.int32]
    name_camadr: numpy.typing.NDArray[numpy.int32]
    name_eqadr: numpy.typing.NDArray[numpy.int32]
    name_excludeadr: numpy.typing.NDArray[numpy.int32]
    name_flexadr: numpy.typing.NDArray[numpy.int32]
    name_geomadr: numpy.typing.NDArray[numpy.int32]
    name_hfieldadr: numpy.typing.NDArray[numpy.int32]
    name_jntadr: numpy.typing.NDArray[numpy.int32]
    name_keyadr: numpy.typing.NDArray[numpy.int32]
    name_lightadr: numpy.typing.NDArray[numpy.int32]
    name_matadr: numpy.typing.NDArray[numpy.int32]
    name_meshadr: numpy.typing.NDArray[numpy.int32]
    name_numericadr: numpy.typing.NDArray[numpy.int32]
    name_pairadr: numpy.typing.NDArray[numpy.int32]
    name_pluginadr: numpy.typing.NDArray[numpy.int32]
    name_sensoradr: numpy.typing.NDArray[numpy.int32]
    name_siteadr: numpy.typing.NDArray[numpy.int32]
    name_skinadr: numpy.typing.NDArray[numpy.int32]
    name_tendonadr: numpy.typing.NDArray[numpy.int32]
    name_texadr: numpy.typing.NDArray[numpy.int32]
    name_textadr: numpy.typing.NDArray[numpy.int32]
    name_tupleadr: numpy.typing.NDArray[numpy.int32]
    names_map: numpy.typing.NDArray[numpy.int32]
    numeric_adr: numpy.typing.NDArray[numpy.int32]
    numeric_data: numpy.typing.NDArray[numpy.float64]
    numeric_size: numpy.typing.NDArray[numpy.int32]
    oct_aabb: numpy.typing.NDArray[numpy.float64]
    oct_child: numpy.typing.NDArray[numpy.int32]
    oct_coeff: numpy.typing.NDArray[numpy.float64]
    oct_depth: numpy.typing.NDArray[numpy.int32]
    pair_dim: numpy.typing.NDArray[numpy.int32]
    pair_friction: numpy.typing.NDArray[numpy.float64]
    pair_gap: numpy.typing.NDArray[numpy.float64]
    pair_geom1: numpy.typing.NDArray[numpy.int32]
    pair_geom2: numpy.typing.NDArray[numpy.int32]
    pair_margin: numpy.typing.NDArray[numpy.float64]
    pair_signature: numpy.typing.NDArray[numpy.int32]
    pair_solimp: numpy.typing.NDArray[numpy.float64]
    pair_solref: numpy.typing.NDArray[numpy.float64]
    pair_solreffriction: numpy.typing.NDArray[numpy.float64]
    plugin: numpy.typing.NDArray[numpy.int32]
    plugin_attr: numpy.typing.NDArray[numpy.int8]
    plugin_attradr: numpy.typing.NDArray[numpy.int32]
    plugin_stateadr: numpy.typing.NDArray[numpy.int32]
    plugin_statenum: numpy.typing.NDArray[numpy.int32]
    qpos0: numpy.typing.NDArray[numpy.float64]
    qpos_spring: numpy.typing.NDArray[numpy.float64]
    sensor_adr: numpy.typing.NDArray[numpy.int32]
    sensor_cutoff: numpy.typing.NDArray[numpy.float64]
    sensor_datatype: numpy.typing.NDArray[numpy.int32]
    sensor_dim: numpy.typing.NDArray[numpy.int32]
    sensor_intprm: numpy.typing.NDArray[numpy.int32]
    sensor_needstage: numpy.typing.NDArray[numpy.int32]
    sensor_noise: numpy.typing.NDArray[numpy.float64]
    sensor_objid: numpy.typing.NDArray[numpy.int32]
    sensor_objtype: numpy.typing.NDArray[numpy.int32]
    sensor_plugin: numpy.typing.NDArray[numpy.int32]
    sensor_refid: numpy.typing.NDArray[numpy.int32]
    sensor_reftype: numpy.typing.NDArray[numpy.int32]
    sensor_type: numpy.typing.NDArray[numpy.int32]
    sensor_user: numpy.typing.NDArray[numpy.float64]
    site_bodyid: numpy.typing.NDArray[numpy.int32]
    site_group: numpy.typing.NDArray[numpy.int32]
    site_matid: numpy.typing.NDArray[numpy.int32]
    site_pos: numpy.typing.NDArray[numpy.float64]
    site_quat: numpy.typing.NDArray[numpy.float64]
    site_rgba: numpy.typing.NDArray[numpy.float32]
    site_sameframe: numpy.typing.NDArray[numpy.uint8]
    site_size: numpy.typing.NDArray[numpy.float64]
    site_type: numpy.typing.NDArray[numpy.int32]
    site_user: numpy.typing.NDArray[numpy.float64]
    skin_boneadr: numpy.typing.NDArray[numpy.int32]
    skin_bonebindpos: numpy.typing.NDArray[numpy.float32]
    skin_bonebindquat: numpy.typing.NDArray[numpy.float32]
    skin_bonebodyid: numpy.typing.NDArray[numpy.int32]
    skin_bonenum: numpy.typing.NDArray[numpy.int32]
    skin_bonevertadr: numpy.typing.NDArray[numpy.int32]
    skin_bonevertid: numpy.typing.NDArray[numpy.int32]
    skin_bonevertnum: numpy.typing.NDArray[numpy.int32]
    skin_bonevertweight: numpy.typing.NDArray[numpy.float32]
    skin_face: numpy.typing.NDArray[numpy.int32]
    skin_faceadr: numpy.typing.NDArray[numpy.int32]
    skin_facenum: numpy.typing.NDArray[numpy.int32]
    skin_group: numpy.typing.NDArray[numpy.int32]
    skin_inflate: numpy.typing.NDArray[numpy.float32]
    skin_matid: numpy.typing.NDArray[numpy.int32]
    skin_pathadr: numpy.typing.NDArray[numpy.int32]
    skin_rgba: numpy.typing.NDArray[numpy.float32]
    skin_texcoord: numpy.typing.NDArray[numpy.float32]
    skin_texcoordadr: numpy.typing.NDArray[numpy.int32]
    skin_vert: numpy.typing.NDArray[numpy.float32]
    skin_vertadr: numpy.typing.NDArray[numpy.int32]
    skin_vertnum: numpy.typing.NDArray[numpy.int32]
    tendon_actfrclimited: numpy.typing.NDArray[numpy.uint8]
    tendon_actfrcrange: numpy.typing.NDArray[numpy.float64]
    tendon_adr: numpy.typing.NDArray[numpy.int32]
    tendon_armature: numpy.typing.NDArray[numpy.float64]
    tendon_damping: numpy.typing.NDArray[numpy.float64]
    tendon_frictionloss: numpy.typing.NDArray[numpy.float64]
    tendon_group: numpy.typing.NDArray[numpy.int32]
    tendon_invweight0: numpy.typing.NDArray[numpy.float64]
    tendon_length0: numpy.typing.NDArray[numpy.float64]
    tendon_lengthspring: numpy.typing.NDArray[numpy.float64]
    tendon_limited: numpy.typing.NDArray[numpy.uint8]
    tendon_margin: numpy.typing.NDArray[numpy.float64]
    tendon_matid: numpy.typing.NDArray[numpy.int32]
    tendon_num: numpy.typing.NDArray[numpy.int32]
    tendon_range: numpy.typing.NDArray[numpy.float64]
    tendon_rgba: numpy.typing.NDArray[numpy.float32]
    tendon_solimp_fri: numpy.typing.NDArray[numpy.float64]
    tendon_solimp_lim: numpy.typing.NDArray[numpy.float64]
    tendon_solref_fri: numpy.typing.NDArray[numpy.float64]
    tendon_solref_lim: numpy.typing.NDArray[numpy.float64]
    tendon_stiffness: numpy.typing.NDArray[numpy.float64]
    tendon_user: numpy.typing.NDArray[numpy.float64]
    tendon_width: numpy.typing.NDArray[numpy.float64]
    tex_adr: numpy.typing.NDArray[numpy.int32]
    tex_colorspace: numpy.typing.NDArray[numpy.int32]
    tex_data: numpy.typing.NDArray[numpy.uint8]
    tex_height: numpy.typing.NDArray[numpy.int32]
    tex_nchannel: numpy.typing.NDArray[numpy.int32]
    tex_pathadr: numpy.typing.NDArray[numpy.int32]
    tex_type: numpy.typing.NDArray[numpy.int32]
    tex_width: numpy.typing.NDArray[numpy.int32]
    text_adr: numpy.typing.NDArray[numpy.int32]
    text_size: numpy.typing.NDArray[numpy.int32]
    tuple_adr: numpy.typing.NDArray[numpy.int32]
    tuple_objid: numpy.typing.NDArray[numpy.int32]
    tuple_objprm: numpy.typing.NDArray[numpy.float64]
    tuple_objtype: numpy.typing.NDArray[numpy.int32]
    tuple_size: numpy.typing.NDArray[numpy.int32]
    wrap_objid: numpy.typing.NDArray[numpy.int32]
    wrap_prm: numpy.typing.NDArray[numpy.float64]
    wrap_type: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def _from_model_ptr(arg0: typing.SupportsInt) -> MjModel:
        """_from_model_ptr(arg0: typing.SupportsInt) -> mujoco._structs.MjModel"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def actuator(self, *args, **kwargs):
        """actuator(*args, **kwargs)
        Overloaded function.

        1. actuator(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelActuatorViews

        2. actuator(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelActuatorViews
        """
    def bind_scalar(self, *args, **kwargs):
        """bind_scalar(*args, **kwargs)
        Overloaded function.

        1. bind_scalar(self: mujoco._structs.MjModel, spec: mjsActuator_ = None) -> mujoco::python::MjModelActuatorViews

        2. bind_scalar(self: mujoco._structs.MjModel, spec: mjsBody_ = None) -> mujoco::python::MjModelBodyViews

        3. bind_scalar(self: mujoco._structs.MjModel, spec: mjsCamera_ = None) -> mujoco::python::MjModelCameraViews

        4. bind_scalar(self: mujoco._structs.MjModel, spec: mjsEquality_ = None) -> mujoco::python::MjModelEqualityViews

        5. bind_scalar(self: mujoco._structs.MjModel, spec: mjsExclude_ = None) -> mujoco::python::MjModelExcludeViews

        6. bind_scalar(self: mujoco._structs.MjModel, spec: mjsGeom_ = None) -> mujoco::python::MjModelGeomViews

        7. bind_scalar(self: mujoco._structs.MjModel, spec: mjsHField_ = None) -> mujoco::python::MjModelHfieldViews

        8. bind_scalar(self: mujoco._structs.MjModel, spec: mjsJoint_ = None) -> mujoco::python::MjModelJointViews

        9. bind_scalar(self: mujoco._structs.MjModel, spec: mjsLight_ = None) -> mujoco::python::MjModelLightViews

        10. bind_scalar(self: mujoco._structs.MjModel, spec: mjsMaterial_ = None) -> mujoco::python::MjModelMaterialViews

        11. bind_scalar(self: mujoco._structs.MjModel, spec: mjsMesh_ = None) -> mujoco::python::MjModelMeshViews

        12. bind_scalar(self: mujoco._structs.MjModel, spec: mjsNumeric_ = None) -> mujoco::python::MjModelNumericViews

        13. bind_scalar(self: mujoco._structs.MjModel, spec: mjsPair_ = None) -> mujoco::python::MjModelPairViews

        14. bind_scalar(self: mujoco._structs.MjModel, spec: mjsSensor_ = None) -> mujoco::python::MjModelSensorViews

        15. bind_scalar(self: mujoco._structs.MjModel, spec: mjsSite_ = None) -> mujoco::python::MjModelSiteViews

        16. bind_scalar(self: mujoco._structs.MjModel, spec: mjsSkin_ = None) -> mujoco::python::MjModelSkinViews

        17. bind_scalar(self: mujoco._structs.MjModel, spec: mjsTendon_ = None) -> mujoco::python::MjModelTendonViews

        18. bind_scalar(self: mujoco._structs.MjModel, spec: mjsTexture_ = None) -> mujoco::python::MjModelTextureViews

        19. bind_scalar(self: mujoco._structs.MjModel, spec: mjsTuple_ = None) -> mujoco::python::MjModelTupleViews

        20. bind_scalar(self: mujoco._structs.MjModel, spec: mjsKey_ = None) -> mujoco::python::MjModelKeyframeViews
        """
    def body(self, *args, **kwargs):
        """body(*args, **kwargs)
        Overloaded function.

        1. body(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelBodyViews

        2. body(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelBodyViews
        """
    def cam(self, *args, **kwargs):
        """cam(*args, **kwargs)
        Overloaded function.

        1. cam(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelCameraViews

        2. cam(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelCameraViews
        """
    def camera(self, *args, **kwargs):
        """camera(*args, **kwargs)
        Overloaded function.

        1. camera(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelCameraViews

        2. camera(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelCameraViews
        """
    def eq(self, *args, **kwargs):
        """eq(*args, **kwargs)
        Overloaded function.

        1. eq(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelEqualityViews

        2. eq(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelEqualityViews
        """
    def equality(self, *args, **kwargs):
        """equality(*args, **kwargs)
        Overloaded function.

        1. equality(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelEqualityViews

        2. equality(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelEqualityViews
        """
    def exclude(self, *args, **kwargs):
        """exclude(*args, **kwargs)
        Overloaded function.

        1. exclude(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelExcludeViews

        2. exclude(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelExcludeViews
        """
    @staticmethod
    def from_binary_path(filename: str, assets: collections.abc.Mapping[str, bytes] | None = ...) -> MjModel:
        """from_binary_path(filename: str, assets: collections.abc.Mapping[str, bytes] | None = None) -> mujoco._structs.MjModel

        Loads an MjModel from an MJB file and an optional assets dictionary.

        The filename for the MJB can also refer to a key in the assets dictionary.
        This is useful for example when the MJB is not available as a file on disk.
        """
    @staticmethod
    def from_xml_path(filename: str, assets: collections.abc.Mapping[str, bytes] | None = ...) -> MjModel:
        """from_xml_path(filename: str, assets: collections.abc.Mapping[str, bytes] | None = None) -> mujoco._structs.MjModel

        Loads an MjModel from an XML file and an optional assets dictionary.

        The filename for the XML can also refer to a key in the assets dictionary.
        This is useful for example when the XML is not available as a file on disk.
        """
    @staticmethod
    def from_xml_string(xml: str, assets: collections.abc.Mapping[str, bytes] | None = ...) -> MjModel:
        """from_xml_string(xml: str, assets: collections.abc.Mapping[str, bytes] | None = None) -> mujoco._structs.MjModel

        Loads an MjModel from an XML string and an optional assets dictionary.
        """
    def geom(self, *args, **kwargs):
        """geom(*args, **kwargs)
        Overloaded function.

        1. geom(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelGeomViews

        2. geom(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelGeomViews
        """
    def hfield(self, *args, **kwargs):
        """hfield(*args, **kwargs)
        Overloaded function.

        1. hfield(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelHfieldViews

        2. hfield(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelHfieldViews
        """
    def jnt(self, *args, **kwargs):
        """jnt(*args, **kwargs)
        Overloaded function.

        1. jnt(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelJointViews

        2. jnt(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelJointViews
        """
    def joint(self, *args, **kwargs):
        """joint(*args, **kwargs)
        Overloaded function.

        1. joint(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelJointViews

        2. joint(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelJointViews
        """
    def key(self, *args, **kwargs):
        """key(*args, **kwargs)
        Overloaded function.

        1. key(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelKeyframeViews

        2. key(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelKeyframeViews
        """
    def keyframe(self, *args, **kwargs):
        """keyframe(*args, **kwargs)
        Overloaded function.

        1. keyframe(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelKeyframeViews

        2. keyframe(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelKeyframeViews
        """
    def light(self, *args, **kwargs):
        """light(*args, **kwargs)
        Overloaded function.

        1. light(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelLightViews

        2. light(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelLightViews
        """
    def mat(self, *args, **kwargs):
        """mat(*args, **kwargs)
        Overloaded function.

        1. mat(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelMaterialViews

        2. mat(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelMaterialViews
        """
    def material(self, *args, **kwargs):
        """material(*args, **kwargs)
        Overloaded function.

        1. material(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelMaterialViews

        2. material(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelMaterialViews
        """
    def mesh(self, *args, **kwargs):
        """mesh(*args, **kwargs)
        Overloaded function.

        1. mesh(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelMeshViews

        2. mesh(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelMeshViews
        """
    def numeric(self, *args, **kwargs):
        """numeric(*args, **kwargs)
        Overloaded function.

        1. numeric(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelNumericViews

        2. numeric(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelNumericViews
        """
    def pair(self, *args, **kwargs):
        """pair(*args, **kwargs)
        Overloaded function.

        1. pair(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelPairViews

        2. pair(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelPairViews
        """
    def sensor(self, *args, **kwargs):
        """sensor(*args, **kwargs)
        Overloaded function.

        1. sensor(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelSensorViews

        2. sensor(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelSensorViews
        """
    def site(self, *args, **kwargs):
        """site(*args, **kwargs)
        Overloaded function.

        1. site(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelSiteViews

        2. site(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelSiteViews
        """
    def skin(self, *args, **kwargs):
        """skin(*args, **kwargs)
        Overloaded function.

        1. skin(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelSkinViews

        2. skin(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelSkinViews
        """
    def tendon(self, *args, **kwargs):
        """tendon(*args, **kwargs)
        Overloaded function.

        1. tendon(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelTendonViews

        2. tendon(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelTendonViews
        """
    def tex(self, *args, **kwargs):
        """tex(*args, **kwargs)
        Overloaded function.

        1. tex(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelTextureViews

        2. tex(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelTextureViews
        """
    def texture(self, *args, **kwargs):
        """texture(*args, **kwargs)
        Overloaded function.

        1. texture(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelTextureViews

        2. texture(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelTextureViews
        """
    def tuple(self, *args, **kwargs):
        """tuple(*args, **kwargs)
        Overloaded function.

        1. tuple(self: mujoco._structs.MjModel, arg0: typing.SupportsInt) -> mujoco::python::MjModelTupleViews

        2. tuple(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelTupleViews
        """
    def __copy__(self) -> MjModel:
        """__copy__(self: mujoco._structs.MjModel) -> mujoco._structs.MjModel"""
    def __deepcopy__(self, arg0: dict) -> MjModel:
        """__deepcopy__(self: mujoco._structs.MjModel, arg0: dict) -> mujoco._structs.MjModel"""
    @property
    def _address(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def _sizes(self) -> numpy.typing.NDArray[numpy.int64]:
        """(arg0: mujoco._structs.MjModel) -> numpy.typing.NDArray[numpy.int64]"""
    @property
    def nB(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nC(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nD(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nJmom(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nM(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def na(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def names(self) -> bytes:
        """(arg0: mujoco._structs.MjModel) -> bytes"""
    @property
    def narena(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nbody(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nbuffer(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nbvh(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nbvhdynamic(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nbvhstatic(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ncam(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nconmax(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nemax(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def neq(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nexclude(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflex(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflexedge(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflexelem(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflexelemdata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflexelemedge(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflexevpair(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflexnode(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflexshelldata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflextexcoord(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nflexvert(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ngeom(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ngravcomp(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nhfield(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nhfielddata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def njmax(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def njnt(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nkey(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nlight(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmat(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmesh(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmeshface(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmeshgraph(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmeshnormal(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmeshpoly(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmeshpolymap(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmeshpolyvert(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmeshtexcoord(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmeshvert(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nmocap(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nnames(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nnames_map(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nnumeric(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nnumericdata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def noct(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def npair(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def npaths(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nplugin(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def npluginattr(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def npluginstate(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nq(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nsensor(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nsensordata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nsite(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nskin(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nskinbone(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nskinbonevert(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nskinface(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nskintexvert(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nskinvert(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ntendon(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ntex(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ntexdata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ntext(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ntextdata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ntree(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ntuple(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def ntupledata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nu(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuser_actuator(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuser_body(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuser_cam(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuser_geom(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuser_jnt(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuser_sensor(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuser_site(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuser_tendon(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nuserdata(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nv(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def nwrap(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def opt(self) -> MjOption:
        """(self: mujoco._structs.MjModel) -> mujoco._structs.MjOption"""
    @property
    def paths(self) -> bytes:
        """(arg0: mujoco._structs.MjModel) -> bytes"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._structs.MjModel) -> int"""
    @property
    def stat(self):
        """(self: mujoco._structs.MjModel) -> mujoco::python::_impl::MjWrapper<mjStatistic_>"""
    @property
    def text_data(self) -> bytes:
        """(arg0: mujoco._structs.MjModel) -> bytes"""
    @property
    def vis(self) -> MjVisual:
        """(self: mujoco._structs.MjModel) -> mujoco._structs.MjVisual"""

class MjOption:
    _float_fields: ClassVar[tuple] = ...  # read-only
    _floatarray_fields: ClassVar[tuple] = ...  # read-only
    _int_fields: ClassVar[tuple] = ...  # read-only
    apirate: float
    ccd_iterations: int
    ccd_tolerance: float
    cone: int
    density: float
    disableactuator: int
    disableflags: int
    enableflags: int
    gravity: numpy.typing.NDArray[numpy.float64]
    impratio: float
    integrator: int
    iterations: int
    jacobian: int
    ls_iterations: int
    ls_tolerance: float
    magnetic: numpy.typing.NDArray[numpy.float64]
    noslip_iterations: int
    noslip_tolerance: float
    o_friction: numpy.typing.NDArray[numpy.float64]
    o_margin: float
    o_solimp: numpy.typing.NDArray[numpy.float64]
    o_solref: numpy.typing.NDArray[numpy.float64]
    sdf_initpoints: int
    sdf_iterations: int
    solver: int
    timestep: float
    tolerance: float
    viscosity: float
    wind: numpy.typing.NDArray[numpy.float64]
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjOption) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjOption:
        """__copy__(self: mujoco._structs.MjOption) -> mujoco._structs.MjOption"""
    def __deepcopy__(self, arg0: dict) -> MjOption:
        """__deepcopy__(self: mujoco._structs.MjOption, arg0: dict) -> mujoco._structs.MjOption"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjSolverStat:
    gradient: float
    improvement: float
    lineslope: float
    nactive: int
    nchange: int
    neval: int
    nupdate: int
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjSolverStat) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjSolverStat:
        """__copy__(self: mujoco._structs.MjSolverStat) -> mujoco._structs.MjSolverStat"""
    def __deepcopy__(self, arg0: dict) -> MjSolverStat:
        """__deepcopy__(self: mujoco._structs.MjSolverStat, arg0: dict) -> mujoco._structs.MjSolverStat"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjStatistic:
    center: numpy.typing.NDArray[numpy.float64]
    extent: float
    meaninertia: float
    meanmass: float
    meansize: float
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjStatistic) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjStatistic:
        """__copy__(self: mujoco._structs.MjStatistic) -> mujoco._structs.MjStatistic"""
    def __deepcopy__(self, arg0: dict) -> MjStatistic:
        """__deepcopy__(self: mujoco._structs.MjStatistic, arg0: dict) -> mujoco._structs.MjStatistic"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjTimerStat:
    duration: float
    number: int
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjTimerStat) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjTimerStat:
        """__copy__(self: mujoco._structs.MjTimerStat) -> mujoco._structs.MjTimerStat"""
    def __deepcopy__(self, arg0: dict) -> MjTimerStat:
        """__deepcopy__(self: mujoco._structs.MjTimerStat, arg0: dict) -> mujoco._structs.MjTimerStat"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjVisual:
    class Global:
        azimuth: float
        bvactive: int
        cameraid: int
        elevation: float
        ellipsoidinertia: int
        fovy: float
        glow: float
        ipd: float
        linewidth: float
        offheight: int
        offwidth: int
        orthographic: int
        realtime: float
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def _pybind11_conduit_v1_(self, *args, **kwargs): ...
        def __copy__(self) -> MjVisual.Global:
            """__copy__(self: mujoco._structs.MjVisual.Global) -> mujoco._structs.MjVisual.Global"""
        def __deepcopy__(self, arg0: dict) -> MjVisual.Global:
            """__deepcopy__(self: mujoco._structs.MjVisual.Global, arg0: dict) -> mujoco._structs.MjVisual.Global"""
        def __eq__(self, arg0: object) -> bool:
            """__eq__(self: object, arg0: object) -> bool"""

    class Headlight:
        active: int
        ambient: numpy.typing.NDArray[numpy.float32]
        diffuse: numpy.typing.NDArray[numpy.float32]
        specular: numpy.typing.NDArray[numpy.float32]
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def _pybind11_conduit_v1_(self, *args, **kwargs): ...
        def __copy__(self) -> MjVisual.Headlight:
            """__copy__(self: mujoco._structs.MjVisual.Headlight) -> mujoco._structs.MjVisual.Headlight"""
        def __deepcopy__(self, arg0: dict) -> MjVisual.Headlight:
            """__deepcopy__(self: mujoco._structs.MjVisual.Headlight, arg0: dict) -> mujoco._structs.MjVisual.Headlight"""
        def __eq__(self, arg0: object) -> bool:
            """__eq__(self: object, arg0: object) -> bool"""

    class Map:
        actuatortendon: float
        alpha: float
        fogend: float
        fogstart: float
        force: float
        haze: float
        shadowclip: float
        shadowscale: float
        stiffness: float
        stiffnessrot: float
        torque: float
        zfar: float
        znear: float
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def _pybind11_conduit_v1_(self, *args, **kwargs): ...
        def __copy__(self) -> MjVisual.Map:
            """__copy__(self: mujoco._structs.MjVisual.Map) -> mujoco._structs.MjVisual.Map"""
        def __deepcopy__(self, arg0: dict) -> MjVisual.Map:
            """__deepcopy__(self: mujoco._structs.MjVisual.Map, arg0: dict) -> mujoco._structs.MjVisual.Map"""
        def __eq__(self, arg0: object) -> bool:
            """__eq__(self: object, arg0: object) -> bool"""

    class Quality:
        numquads: int
        numslices: int
        numstacks: int
        offsamples: int
        shadowsize: int
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def _pybind11_conduit_v1_(self, *args, **kwargs): ...
        def __copy__(self) -> MjVisual.Quality:
            """__copy__(self: mujoco._structs.MjVisual.Quality) -> mujoco._structs.MjVisual.Quality"""
        def __deepcopy__(self, arg0: dict) -> MjVisual.Quality:
            """__deepcopy__(self: mujoco._structs.MjVisual.Quality, arg0: dict) -> mujoco._structs.MjVisual.Quality"""
        def __eq__(self, arg0: object) -> bool:
            """__eq__(self: object, arg0: object) -> bool"""

    class Rgba:
        actuator: numpy.typing.NDArray[numpy.float32]
        actuatornegative: numpy.typing.NDArray[numpy.float32]
        actuatorpositive: numpy.typing.NDArray[numpy.float32]
        bv: numpy.typing.NDArray[numpy.float32]
        bvactive: numpy.typing.NDArray[numpy.float32]
        camera: numpy.typing.NDArray[numpy.float32]
        com: numpy.typing.NDArray[numpy.float32]
        connect: numpy.typing.NDArray[numpy.float32]
        constraint: numpy.typing.NDArray[numpy.float32]
        contactforce: numpy.typing.NDArray[numpy.float32]
        contactfriction: numpy.typing.NDArray[numpy.float32]
        contactgap: numpy.typing.NDArray[numpy.float32]
        contactpoint: numpy.typing.NDArray[numpy.float32]
        contacttorque: numpy.typing.NDArray[numpy.float32]
        crankbroken: numpy.typing.NDArray[numpy.float32]
        fog: numpy.typing.NDArray[numpy.float32]
        force: numpy.typing.NDArray[numpy.float32]
        frustum: numpy.typing.NDArray[numpy.float32]
        haze: numpy.typing.NDArray[numpy.float32]
        inertia: numpy.typing.NDArray[numpy.float32]
        joint: numpy.typing.NDArray[numpy.float32]
        light: numpy.typing.NDArray[numpy.float32]
        rangefinder: numpy.typing.NDArray[numpy.float32]
        selectpoint: numpy.typing.NDArray[numpy.float32]
        slidercrank: numpy.typing.NDArray[numpy.float32]
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def _pybind11_conduit_v1_(self, *args, **kwargs): ...
        def __copy__(self) -> MjVisual.Rgba:
            """__copy__(self: mujoco._structs.MjVisual.Rgba) -> mujoco._structs.MjVisual.Rgba"""
        def __deepcopy__(self, arg0: dict) -> MjVisual.Rgba:
            """__deepcopy__(self: mujoco._structs.MjVisual.Rgba, arg0: dict) -> mujoco._structs.MjVisual.Rgba"""
        def __eq__(self, arg0: object) -> bool:
            """__eq__(self: object, arg0: object) -> bool"""

    class Scale:
        actuatorlength: float
        actuatorwidth: float
        camera: float
        com: float
        connect: float
        constraint: float
        contactheight: float
        contactwidth: float
        forcewidth: float
        framelength: float
        framewidth: float
        frustum: float
        jointlength: float
        jointwidth: float
        light: float
        selectpoint: float
        slidercrank: float
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def _pybind11_conduit_v1_(self, *args, **kwargs): ...
        def __copy__(self) -> MjVisual.Scale:
            """__copy__(self: mujoco._structs.MjVisual.Scale) -> mujoco._structs.MjVisual.Scale"""
        def __deepcopy__(self, arg0: dict) -> MjVisual.Scale:
            """__deepcopy__(self: mujoco._structs.MjVisual.Scale, arg0: dict) -> mujoco._structs.MjVisual.Scale"""
        def __eq__(self, arg0: object) -> bool:
            """__eq__(self: object, arg0: object) -> bool"""
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjVisual:
        """__copy__(self: mujoco._structs.MjVisual) -> mujoco._structs.MjVisual"""
    def __deepcopy__(self, arg0: dict) -> MjVisual:
        """__deepcopy__(self: mujoco._structs.MjVisual, arg0: dict) -> mujoco._structs.MjVisual"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @property
    def global_(self) -> MjVisual.Global:
        """(arg0: mujoco._structs.MjVisual) -> mujoco._structs.MjVisual.Global"""
    @property
    def headlight(self) -> MjVisual.Headlight:
        """(self: mujoco._structs.MjVisual) -> mujoco._structs.MjVisual.Headlight"""
    @property
    def map(self) -> MjVisual.Map:
        """(arg0: mujoco._structs.MjVisual) -> mujoco._structs.MjVisual.Map"""
    @property
    def quality(self) -> MjVisual.Quality:
        """(arg0: mujoco._structs.MjVisual) -> mujoco._structs.MjVisual.Quality"""
    @property
    def rgba(self) -> MjVisual.Rgba:
        """(self: mujoco._structs.MjVisual) -> mujoco._structs.MjVisual.Rgba"""
    @property
    def scale(self) -> MjVisual.Scale:
        """(arg0: mujoco._structs.MjVisual) -> mujoco._structs.MjVisual.Scale"""

class MjWarningStat:
    lastinfo: int
    number: int
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjWarningStat) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjWarningStat:
        """__copy__(self: mujoco._structs.MjWarningStat) -> mujoco._structs.MjWarningStat"""
    def __deepcopy__(self, arg0: dict) -> MjWarningStat:
        """__deepcopy__(self: mujoco._structs.MjWarningStat, arg0: dict) -> mujoco._structs.MjWarningStat"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjvCamera:
    azimuth: float
    distance: float
    elevation: float
    fixedcamid: int
    lookat: numpy.typing.NDArray[numpy.float64]
    orthographic: int
    trackbodyid: int
    type: int
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjvCamera) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvCamera:
        """__copy__(self: mujoco._structs.MjvCamera) -> mujoco._structs.MjvCamera"""
    def __deepcopy__(self, arg0: dict) -> MjvCamera:
        """__deepcopy__(self: mujoco._structs.MjvCamera, arg0: dict) -> mujoco._structs.MjvCamera"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjvFigure:
    figurergba: numpy.typing.NDArray[numpy.float32]
    flg_barplot: int
    flg_extend: int
    flg_legend: int
    flg_selection: int
    flg_symmetric: int
    flg_ticklabel: numpy.typing.NDArray[numpy.int32]
    gridrgb: numpy.typing.NDArray[numpy.float32]
    gridsize: numpy.typing.NDArray[numpy.int32]
    gridwidth: float
    highlight: numpy.typing.NDArray[numpy.int32]
    highlightid: int
    legendoffset: int
    legendrgba: numpy.typing.NDArray[numpy.float32]
    linedata: numpy.typing.NDArray[numpy.float32]
    linepnt: numpy.typing.NDArray[numpy.int32]
    linergb: numpy.typing.NDArray[numpy.float32]
    linewidth: float
    minwidth: str
    panergba: numpy.typing.NDArray[numpy.float32]
    range: numpy.typing.NDArray[numpy.float32]
    selection: float
    subplot: int
    textrgb: numpy.typing.NDArray[numpy.float32]
    title: str
    xaxisdata: numpy.typing.NDArray[numpy.float32]
    xaxispixel: numpy.typing.NDArray[numpy.int32]
    xformat: str
    xlabel: str
    yaxisdata: numpy.typing.NDArray[numpy.float32]
    yaxispixel: numpy.typing.NDArray[numpy.int32]
    yformat: str
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjvFigure) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvFigure:
        """__copy__(self: mujoco._structs.MjvFigure) -> mujoco._structs.MjvFigure"""
    def __deepcopy__(self, arg0: dict) -> MjvFigure:
        """__deepcopy__(self: mujoco._structs.MjvFigure, arg0: dict) -> mujoco._structs.MjvFigure"""
    @property
    def linename(self) -> numpy.ndarray:
        """(self: mujoco._structs.MjvFigure) -> numpy.ndarray"""

class MjvGLCamera:
    forward: numpy.typing.NDArray[numpy.float32]
    frustum_bottom: float
    frustum_center: float
    frustum_far: float
    frustum_near: float
    frustum_top: float
    frustum_width: float
    orthographic: int
    pos: numpy.typing.NDArray[numpy.float32]
    up: numpy.typing.NDArray[numpy.float32]
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjvGLCamera) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvGLCamera:
        """__copy__(self: mujoco._structs.MjvGLCamera) -> mujoco._structs.MjvGLCamera"""
    def __deepcopy__(self, arg0: dict) -> MjvGLCamera:
        """__deepcopy__(self: mujoco._structs.MjvGLCamera, arg0: dict) -> mujoco._structs.MjvGLCamera"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjvGeom:
    camdist: float
    category: int
    dataid: int
    emission: float
    label: str
    mat: numpy.typing.NDArray[numpy.float32]
    matid: int
    modelrbound: float
    objid: int
    objtype: int
    pos: numpy.typing.NDArray[numpy.float32]
    reflectance: float
    rgba: numpy.typing.NDArray[numpy.float32]
    segid: int
    shininess: float
    size: numpy.typing.NDArray[numpy.float32]
    specular: float
    texcoord: int
    transparent: int
    type: int
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjvGeom) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvGeom:
        """__copy__(self: mujoco._structs.MjvGeom) -> mujoco._structs.MjvGeom"""
    def __deepcopy__(self, arg0: dict) -> MjvGeom:
        """__deepcopy__(self: mujoco._structs.MjvGeom, arg0: dict) -> mujoco._structs.MjvGeom"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjvLight:
    ambient: numpy.typing.NDArray[numpy.float32]
    attenuation: numpy.typing.NDArray[numpy.float32]
    bulbradius: float
    castshadow: int
    cutoff: float
    diffuse: numpy.typing.NDArray[numpy.float32]
    dir: numpy.typing.NDArray[numpy.float32]
    exponent: float
    headlight: int
    id: int
    intensity: float
    pos: numpy.typing.NDArray[numpy.float32]
    range: float
    specular: numpy.typing.NDArray[numpy.float32]
    texid: int
    type: int
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjvLight) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvLight:
        """__copy__(self: mujoco._structs.MjvLight) -> mujoco._structs.MjvLight"""
    def __deepcopy__(self, arg0: dict) -> MjvLight:
        """__deepcopy__(self: mujoco._structs.MjvLight, arg0: dict) -> mujoco._structs.MjvLight"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjvOption:
    actuatorgroup: numpy.typing.NDArray[numpy.uint8]
    bvh_depth: int
    flags: numpy.typing.NDArray[numpy.uint8]
    flex_layer: int
    flexgroup: numpy.typing.NDArray[numpy.uint8]
    frame: int
    geomgroup: numpy.typing.NDArray[numpy.uint8]
    jointgroup: numpy.typing.NDArray[numpy.uint8]
    label: int
    sitegroup: numpy.typing.NDArray[numpy.uint8]
    skingroup: numpy.typing.NDArray[numpy.uint8]
    tendongroup: numpy.typing.NDArray[numpy.uint8]
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjvOption) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvOption:
        """__copy__(self: mujoco._structs.MjvOption) -> mujoco._structs.MjvOption"""
    def __deepcopy__(self, arg0: dict) -> MjvOption:
        """__deepcopy__(self: mujoco._structs.MjvOption, arg0: dict) -> mujoco._structs.MjvOption"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjvPerturb:
    active: int
    active2: int
    flexselect: int
    localmass: float
    localpos: numpy.typing.NDArray[numpy.float64]
    refpos: numpy.typing.NDArray[numpy.float64]
    refquat: numpy.typing.NDArray[numpy.float64]
    refselpos: numpy.typing.NDArray[numpy.float64]
    scale: float
    select: int
    skinselect: int
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjvPerturb) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvPerturb:
        """__copy__(self: mujoco._structs.MjvPerturb) -> mujoco._structs.MjvPerturb"""
    def __deepcopy__(self, arg0: dict) -> MjvPerturb:
        """__deepcopy__(self: mujoco._structs.MjvPerturb, arg0: dict) -> mujoco._structs.MjvPerturb"""
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""

class MjvScene:
    enabletransform: int
    flags: numpy.typing.NDArray[numpy.uint8]
    flexedge: numpy.typing.NDArray[numpy.int32]
    flexedgeadr: numpy.typing.NDArray[numpy.int32]
    flexedgenum: numpy.typing.NDArray[numpy.int32]
    flexedgeopt: int
    flexface: numpy.typing.NDArray[numpy.float32]
    flexfaceadr: numpy.typing.NDArray[numpy.int32]
    flexfacenum: numpy.typing.NDArray[numpy.int32]
    flexfaceopt: int
    flexfaceused: numpy.typing.NDArray[numpy.int32]
    flexnormal: numpy.typing.NDArray[numpy.float32]
    flexskinopt: int
    flextexcoord: numpy.typing.NDArray[numpy.float32]
    flexvert: numpy.typing.NDArray[numpy.float32]
    flexvertadr: numpy.typing.NDArray[numpy.int32]
    flexvertnum: numpy.typing.NDArray[numpy.int32]
    flexvertopt: int
    framergb: numpy.typing.NDArray[numpy.float32]
    framewidth: int
    geomorder: numpy.typing.NDArray[numpy.int32]
    maxgeom: int
    nflex: int
    ngeom: int
    nlight: int
    nskin: int
    rotate: numpy.typing.NDArray[numpy.float32]
    scale: float
    skinfacenum: numpy.typing.NDArray[numpy.int32]
    skinnormal: numpy.typing.NDArray[numpy.float32]
    skinvert: numpy.typing.NDArray[numpy.float32]
    skinvertadr: numpy.typing.NDArray[numpy.int32]
    skinvertnum: numpy.typing.NDArray[numpy.int32]
    stereo: int
    translate: numpy.typing.NDArray[numpy.float32]
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._structs.MjvScene) -> None

        2. __init__(self: mujoco._structs.MjvScene, model: mujoco._structs.MjModel, maxgeom: typing.SupportsInt) -> None
        """
    @overload
    def __init__(self, model: MjModel, maxgeom: typing.SupportsInt) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._structs.MjvScene) -> None

        2. __init__(self: mujoco._structs.MjvScene, model: mujoco._structs.MjModel, maxgeom: typing.SupportsInt) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvScene:
        """__copy__(self: mujoco._structs.MjvScene) -> mujoco._structs.MjvScene"""
    def __deepcopy__(self, arg0: dict) -> MjvScene:
        """__deepcopy__(self: mujoco._structs.MjvScene, arg0: dict) -> mujoco._structs.MjvScene"""
    @property
    def camera(self) -> tuple:
        """(arg0: mujoco._structs.MjvScene) -> tuple"""
    @property
    def geoms(self) -> tuple:
        """(arg0: mujoco._structs.MjvScene) -> tuple"""
    @property
    def lights(self) -> tuple:
        """(arg0: mujoco._structs.MjvScene) -> tuple"""

class _MjContactList:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @overload
    def __getitem__(self, arg0: typing.SupportsInt) -> MjContact:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjContactList, arg0: typing.SupportsInt) -> mujoco._structs.MjContact

        2. __getitem__(self: mujoco._structs._MjContactList, arg0: slice) -> mujoco._structs._MjContactList
        """
    @overload
    def __getitem__(self, arg0: slice) -> _MjContactList:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjContactList, arg0: typing.SupportsInt) -> mujoco._structs.MjContact

        2. __getitem__(self: mujoco._structs._MjContactList, arg0: slice) -> mujoco._structs._MjContactList
        """
    def __len__(self) -> int:
        """__len__(self: mujoco._structs._MjContactList) -> int"""
    @property
    def H(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def dim(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def dist(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def efc_address(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def elem(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def exclude(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def flex(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def frame(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def friction(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def geom(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def geom1(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def geom2(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def includemargin(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def mu(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def pos(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def solimp(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def solref(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def solreffriction(self) -> numpy.typing.NDArray[numpy.float64]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def vert(self) -> numpy.typing.NDArray[numpy.int32]:
        """(arg0: mujoco._structs._MjContactList) -> numpy.typing.NDArray[numpy.int32]"""

class _MjDataActuatorViews:
    ctrl: numpy.typing.NDArray[numpy.float64]
    force: numpy.typing.NDArray[numpy.float64]
    length: numpy.typing.NDArray[numpy.float64]
    moment: numpy.typing.NDArray[numpy.float64]
    velocity: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataActuatorViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataActuatorViews) -> str"""

class _MjDataBodyViews:
    cacc: numpy.typing.NDArray[numpy.float64]
    cfrc_ext: numpy.typing.NDArray[numpy.float64]
    cfrc_int: numpy.typing.NDArray[numpy.float64]
    cinert: numpy.typing.NDArray[numpy.float64]
    crb: numpy.typing.NDArray[numpy.float64]
    cvel: numpy.typing.NDArray[numpy.float64]
    subtree_angmom: numpy.typing.NDArray[numpy.float64]
    subtree_com: numpy.typing.NDArray[numpy.float64]
    subtree_linvel: numpy.typing.NDArray[numpy.float64]
    xfrc_applied: numpy.typing.NDArray[numpy.float64]
    ximat: numpy.typing.NDArray[numpy.float64]
    xipos: numpy.typing.NDArray[numpy.float64]
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    xquat: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataBodyViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataBodyViews) -> str"""

class _MjDataCameraViews:
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataCameraViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataCameraViews) -> str"""

class _MjDataGeomViews:
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataGeomViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataGeomViews) -> str"""

class _MjDataJointViews:
    cdof: numpy.typing.NDArray[numpy.float64]
    cdof_dot: numpy.typing.NDArray[numpy.float64]
    qLDiagInv: numpy.typing.NDArray[numpy.float64]
    qacc: numpy.typing.NDArray[numpy.float64]
    qacc_smooth: numpy.typing.NDArray[numpy.float64]
    qacc_warmstart: numpy.typing.NDArray[numpy.float64]
    qfrc_actuator: numpy.typing.NDArray[numpy.float64]
    qfrc_applied: numpy.typing.NDArray[numpy.float64]
    qfrc_bias: numpy.typing.NDArray[numpy.float64]
    qfrc_constraint: numpy.typing.NDArray[numpy.float64]
    qfrc_inverse: numpy.typing.NDArray[numpy.float64]
    qfrc_passive: numpy.typing.NDArray[numpy.float64]
    qfrc_smooth: numpy.typing.NDArray[numpy.float64]
    qpos: numpy.typing.NDArray[numpy.float64]
    qvel: numpy.typing.NDArray[numpy.float64]
    xanchor: numpy.typing.NDArray[numpy.float64]
    xaxis: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataJointViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataJointViews) -> str"""

class _MjDataLightViews:
    xdir: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataLightViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataLightViews) -> str"""

class _MjDataSensorViews:
    data: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataSensorViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataSensorViews) -> str"""

class _MjDataSiteViews:
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataSiteViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataSiteViews) -> str"""

class _MjDataTendonViews:
    J: numpy.typing.NDArray[numpy.float64]
    J_colind: numpy.typing.NDArray[numpy.int32]
    J_rowadr: numpy.typing.NDArray[numpy.int32]
    J_rownnz: numpy.typing.NDArray[numpy.int32]
    length: numpy.typing.NDArray[numpy.float64]
    velocity: numpy.typing.NDArray[numpy.float64]
    wrapadr: numpy.typing.NDArray[numpy.int32]
    wrapnum: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjDataTendonViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjDataTendonViews) -> str"""

class _MjModelActuatorViews:
    acc0: numpy.typing.NDArray[numpy.float64]
    actadr: numpy.typing.NDArray[numpy.int32]
    actlimited: numpy.typing.NDArray[numpy.uint8]
    actnum: numpy.typing.NDArray[numpy.int32]
    actrange: numpy.typing.NDArray[numpy.float64]
    biasprm: numpy.typing.NDArray[numpy.float64]
    biastype: numpy.typing.NDArray[numpy.int32]
    cranklength: numpy.typing.NDArray[numpy.float64]
    ctrllimited: numpy.typing.NDArray[numpy.uint8]
    ctrlrange: numpy.typing.NDArray[numpy.float64]
    dynprm: numpy.typing.NDArray[numpy.float64]
    dyntype: numpy.typing.NDArray[numpy.int32]
    forcelimited: numpy.typing.NDArray[numpy.uint8]
    forcerange: numpy.typing.NDArray[numpy.float64]
    gainprm: numpy.typing.NDArray[numpy.float64]
    gaintype: numpy.typing.NDArray[numpy.int32]
    gear: numpy.typing.NDArray[numpy.float64]
    group: numpy.typing.NDArray[numpy.int32]
    length0: numpy.typing.NDArray[numpy.float64]
    lengthrange: numpy.typing.NDArray[numpy.float64]
    trnid: numpy.typing.NDArray[numpy.int32]
    trntype: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelActuatorViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelActuatorViews) -> str"""

class _MjModelBodyViews:
    dofadr: numpy.typing.NDArray[numpy.int32]
    dofnum: numpy.typing.NDArray[numpy.int32]
    geomadr: numpy.typing.NDArray[numpy.int32]
    geomnum: numpy.typing.NDArray[numpy.int32]
    inertia: numpy.typing.NDArray[numpy.float64]
    invweight0: numpy.typing.NDArray[numpy.float64]
    ipos: numpy.typing.NDArray[numpy.float64]
    iquat: numpy.typing.NDArray[numpy.float64]
    jntadr: numpy.typing.NDArray[numpy.int32]
    jntnum: numpy.typing.NDArray[numpy.int32]
    mass: numpy.typing.NDArray[numpy.float64]
    mocapid: numpy.typing.NDArray[numpy.int32]
    parentid: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    quat: numpy.typing.NDArray[numpy.float64]
    rootid: numpy.typing.NDArray[numpy.int32]
    sameframe: numpy.typing.NDArray[numpy.uint8]
    simple: numpy.typing.NDArray[numpy.uint8]
    subtreemass: numpy.typing.NDArray[numpy.float64]
    user: numpy.typing.NDArray[numpy.float64]
    weldid: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelBodyViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelBodyViews) -> str"""

class _MjModelCameraViews:
    bodyid: numpy.typing.NDArray[numpy.int32]
    fovy: numpy.typing.NDArray[numpy.float64]
    ipd: numpy.typing.NDArray[numpy.float64]
    mat0: numpy.typing.NDArray[numpy.float64]
    mode: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    pos0: numpy.typing.NDArray[numpy.float64]
    poscom0: numpy.typing.NDArray[numpy.float64]
    quat: numpy.typing.NDArray[numpy.float64]
    targetbodyid: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelCameraViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelCameraViews) -> str"""

class _MjModelEqualityViews:
    active0: numpy.typing.NDArray[numpy.uint8]
    data: numpy.typing.NDArray[numpy.float64]
    obj1id: numpy.typing.NDArray[numpy.int32]
    obj2id: numpy.typing.NDArray[numpy.int32]
    solimp: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    type: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelEqualityViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelEqualityViews) -> str"""

class _MjModelExcludeViews:
    signature: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelExcludeViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelExcludeViews) -> str"""

class _MjModelGeomViews:
    bodyid: numpy.typing.NDArray[numpy.int32]
    conaffinity: numpy.typing.NDArray[numpy.int32]
    condim: numpy.typing.NDArray[numpy.int32]
    contype: numpy.typing.NDArray[numpy.int32]
    dataid: numpy.typing.NDArray[numpy.int32]
    friction: numpy.typing.NDArray[numpy.float64]
    gap: numpy.typing.NDArray[numpy.float64]
    group: numpy.typing.NDArray[numpy.int32]
    margin: numpy.typing.NDArray[numpy.float64]
    matid: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    priority: numpy.typing.NDArray[numpy.int32]
    quat: numpy.typing.NDArray[numpy.float64]
    rbound: numpy.typing.NDArray[numpy.float64]
    rgba: numpy.typing.NDArray[numpy.float32]
    sameframe: numpy.typing.NDArray[numpy.uint8]
    size: numpy.typing.NDArray[numpy.float64]
    solimp: numpy.typing.NDArray[numpy.float64]
    solmix: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    type: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelGeomViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelGeomViews) -> str"""

class _MjModelHfieldViews:
    adr: numpy.typing.NDArray[numpy.int32]
    data: numpy.typing.NDArray[numpy.float32]
    ncol: numpy.typing.NDArray[numpy.int32]
    nrow: numpy.typing.NDArray[numpy.int32]
    size: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelHfieldViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelHfieldViews) -> str"""

class _MjModelJointViews:
    M0: numpy.typing.NDArray[numpy.float64]
    Madr: numpy.typing.NDArray[numpy.int32]
    armature: numpy.typing.NDArray[numpy.float64]
    axis: numpy.typing.NDArray[numpy.float64]
    bodyid: numpy.typing.NDArray[numpy.int32]
    damping: numpy.typing.NDArray[numpy.float64]
    dofadr: numpy.typing.NDArray[numpy.int32]
    frictionloss: numpy.typing.NDArray[numpy.float64]
    group: numpy.typing.NDArray[numpy.int32]
    invweight0: numpy.typing.NDArray[numpy.float64]
    jntid: numpy.typing.NDArray[numpy.int32]
    limited: numpy.typing.NDArray[numpy.uint8]
    margin: numpy.typing.NDArray[numpy.float64]
    parentid: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    qpos0: numpy.typing.NDArray[numpy.float64]
    qpos_spring: numpy.typing.NDArray[numpy.float64]
    qposadr: numpy.typing.NDArray[numpy.int32]
    range: numpy.typing.NDArray[numpy.float64]
    simplenum: numpy.typing.NDArray[numpy.int32]
    solimp: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    stiffness: numpy.typing.NDArray[numpy.float64]
    type: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelJointViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelJointViews) -> str"""

class _MjModelKeyframeViews:
    act: numpy.typing.NDArray[numpy.float64]
    ctrl: numpy.typing.NDArray[numpy.float64]
    mpos: numpy.typing.NDArray[numpy.float64]
    mquat: numpy.typing.NDArray[numpy.float64]
    qpos: numpy.typing.NDArray[numpy.float64]
    qvel: numpy.typing.NDArray[numpy.float64]
    time: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelKeyframeViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelKeyframeViews) -> str"""

class _MjModelLightViews:
    active: numpy.typing.NDArray[numpy.uint8]
    ambient: numpy.typing.NDArray[numpy.float32]
    attenuation: numpy.typing.NDArray[numpy.float32]
    bodyid: numpy.typing.NDArray[numpy.int32]
    castshadow: numpy.typing.NDArray[numpy.uint8]
    cutoff: numpy.typing.NDArray[numpy.float32]
    diffuse: numpy.typing.NDArray[numpy.float32]
    dir: numpy.typing.NDArray[numpy.float64]
    dir0: numpy.typing.NDArray[numpy.float64]
    exponent: numpy.typing.NDArray[numpy.float32]
    mode: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    pos0: numpy.typing.NDArray[numpy.float64]
    poscom0: numpy.typing.NDArray[numpy.float64]
    specular: numpy.typing.NDArray[numpy.float32]
    targetbodyid: numpy.typing.NDArray[numpy.int32]
    type: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelLightViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelLightViews) -> str"""

class _MjModelMaterialViews:
    emission: numpy.typing.NDArray[numpy.float32]
    reflectance: numpy.typing.NDArray[numpy.float32]
    rgba: numpy.typing.NDArray[numpy.float32]
    shininess: numpy.typing.NDArray[numpy.float32]
    specular: numpy.typing.NDArray[numpy.float32]
    texid: numpy.typing.NDArray[numpy.int32]
    texrepeat: numpy.typing.NDArray[numpy.float32]
    texuniform: numpy.typing.NDArray[numpy.uint8]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelMaterialViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelMaterialViews) -> str"""

class _MjModelMeshViews:
    faceadr: numpy.typing.NDArray[numpy.int32]
    facenum: numpy.typing.NDArray[numpy.int32]
    graphadr: numpy.typing.NDArray[numpy.int32]
    texcoordadr: numpy.typing.NDArray[numpy.int32]
    vertadr: numpy.typing.NDArray[numpy.int32]
    vertnum: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelMeshViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelMeshViews) -> str"""

class _MjModelNumericViews:
    adr: numpy.typing.NDArray[numpy.int32]
    data: numpy.typing.NDArray[numpy.float64]
    size: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelNumericViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelNumericViews) -> str"""

class _MjModelPairViews:
    dim: numpy.typing.NDArray[numpy.int32]
    friction: numpy.typing.NDArray[numpy.float64]
    gap: numpy.typing.NDArray[numpy.float64]
    geom1: numpy.typing.NDArray[numpy.int32]
    geom2: numpy.typing.NDArray[numpy.int32]
    margin: numpy.typing.NDArray[numpy.float64]
    signature: numpy.typing.NDArray[numpy.int32]
    solimp: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelPairViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelPairViews) -> str"""

class _MjModelSensorViews:
    adr: numpy.typing.NDArray[numpy.int32]
    cutoff: numpy.typing.NDArray[numpy.float64]
    datatype: numpy.typing.NDArray[numpy.int32]
    dim: numpy.typing.NDArray[numpy.int32]
    needstage: numpy.typing.NDArray[numpy.int32]
    noise: numpy.typing.NDArray[numpy.float64]
    objid: numpy.typing.NDArray[numpy.int32]
    objtype: numpy.typing.NDArray[numpy.int32]
    refid: numpy.typing.NDArray[numpy.int32]
    reftype: numpy.typing.NDArray[numpy.int32]
    type: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelSensorViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelSensorViews) -> str"""

class _MjModelSiteViews:
    bodyid: numpy.typing.NDArray[numpy.int32]
    group: numpy.typing.NDArray[numpy.int32]
    matid: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    quat: numpy.typing.NDArray[numpy.float64]
    rgba: numpy.typing.NDArray[numpy.float32]
    sameframe: numpy.typing.NDArray[numpy.uint8]
    size: numpy.typing.NDArray[numpy.float64]
    type: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelSiteViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelSiteViews) -> str"""

class _MjModelSkinViews:
    boneadr: numpy.typing.NDArray[numpy.int32]
    bonenum: numpy.typing.NDArray[numpy.int32]
    faceadr: numpy.typing.NDArray[numpy.int32]
    facenum: numpy.typing.NDArray[numpy.int32]
    inflate: numpy.typing.NDArray[numpy.float32]
    matid: numpy.typing.NDArray[numpy.int32]
    rgba: numpy.typing.NDArray[numpy.float32]
    texcoordadr: numpy.typing.NDArray[numpy.int32]
    vertadr: numpy.typing.NDArray[numpy.int32]
    vertnum: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelSkinViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelSkinViews) -> str"""

class _MjModelTendonViews:
    _adr: numpy.typing.NDArray[numpy.int32]
    _damping: numpy.typing.NDArray[numpy.float64]
    _frictionloss: numpy.typing.NDArray[numpy.float64]
    _group: numpy.typing.NDArray[numpy.int32]
    _invweight0: numpy.typing.NDArray[numpy.float64]
    _length0: numpy.typing.NDArray[numpy.float64]
    _lengthspring: numpy.typing.NDArray[numpy.float64]
    _limited: numpy.typing.NDArray[numpy.uint8]
    _margin: numpy.typing.NDArray[numpy.float64]
    _matid: numpy.typing.NDArray[numpy.int32]
    _num: numpy.typing.NDArray[numpy.int32]
    _range: numpy.typing.NDArray[numpy.float64]
    _rgba: numpy.typing.NDArray[numpy.float32]
    _solimp_fri: numpy.typing.NDArray[numpy.float64]
    _solimp_lim: numpy.typing.NDArray[numpy.float64]
    _solref_fri: numpy.typing.NDArray[numpy.float64]
    _solref_lim: numpy.typing.NDArray[numpy.float64]
    _stiffness: numpy.typing.NDArray[numpy.float64]
    _user: numpy.typing.NDArray[numpy.float64]
    _width: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelTendonViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelTendonViews) -> str"""

class _MjModelTextureViews:
    adr: numpy.typing.NDArray[numpy.int32]
    data: numpy.typing.NDArray[numpy.uint8]
    height: numpy.typing.NDArray[numpy.int32]
    nchannel: numpy.typing.NDArray[numpy.int32]
    type: numpy.typing.NDArray[numpy.int32]
    width: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelTextureViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelTextureViews) -> str"""

class _MjModelTupleViews:
    adr: numpy.typing.NDArray[numpy.int32]
    objid: numpy.typing.NDArray[numpy.int32]
    objprm: numpy.typing.NDArray[numpy.float64]
    objtype: numpy.typing.NDArray[numpy.int32]
    size: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._structs._MjModelTupleViews) -> int"""
    @property
    def name(self) -> str:
        """(arg0: mujoco._structs._MjModelTupleViews) -> str"""

class _MjSolverStatList:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @overload
    def __getitem__(self, arg0: typing.SupportsInt) -> MjSolverStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjSolverStatList, arg0: typing.SupportsInt) -> mujoco._structs.MjSolverStat

        2. __getitem__(self: mujoco._structs._MjSolverStatList, arg0: slice) -> mujoco._structs._MjSolverStatList
        """
    @overload
    def __getitem__(self, arg0: slice) -> _MjSolverStatList:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjSolverStatList, arg0: typing.SupportsInt) -> mujoco._structs.MjSolverStat

        2. __getitem__(self: mujoco._structs._MjSolverStatList, arg0: slice) -> mujoco._structs._MjSolverStatList
        """
    def __len__(self) -> int:
        """__len__(self: mujoco._structs._MjSolverStatList) -> int"""
    @property
    def gradient(self) -> numpy.typing.NDArray[numpy.float64]:
        """(self: mujoco._structs._MjSolverStatList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def improvement(self) -> numpy.typing.NDArray[numpy.float64]:
        """(self: mujoco._structs._MjSolverStatList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def lineslope(self) -> numpy.typing.NDArray[numpy.float64]:
        """(self: mujoco._structs._MjSolverStatList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def nactive(self) -> numpy.typing.NDArray[numpy.int32]:
        """(self: mujoco._structs._MjSolverStatList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def nchange(self) -> numpy.typing.NDArray[numpy.int32]:
        """(self: mujoco._structs._MjSolverStatList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def neval(self) -> numpy.typing.NDArray[numpy.int32]:
        """(self: mujoco._structs._MjSolverStatList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def nupdate(self) -> numpy.typing.NDArray[numpy.int32]:
        """(self: mujoco._structs._MjSolverStatList) -> numpy.typing.NDArray[numpy.int32]"""

class _MjTimerStatList:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @overload
    def __getitem__(self, arg0: typing.SupportsInt) -> MjTimerStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: typing.SupportsInt) -> mujoco._structs.MjTimerStat

        2. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: mujoco._enums.mjtTimer) -> mujoco._structs.MjTimerStat

        3. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: slice) -> mujoco._structs._MjTimerStatList
        """
    @overload
    def __getitem__(self, arg0: mujoco._enums.mjtTimer) -> MjTimerStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: typing.SupportsInt) -> mujoco._structs.MjTimerStat

        2. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: mujoco._enums.mjtTimer) -> mujoco._structs.MjTimerStat

        3. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: slice) -> mujoco._structs._MjTimerStatList
        """
    @overload
    def __getitem__(self, arg0: slice) -> _MjTimerStatList:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: typing.SupportsInt) -> mujoco._structs.MjTimerStat

        2. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: mujoco._enums.mjtTimer) -> mujoco._structs.MjTimerStat

        3. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: slice) -> mujoco._structs._MjTimerStatList
        """
    def __len__(self) -> int:
        """__len__(self: mujoco._structs._MjTimerStatList) -> int"""
    @property
    def duration(self) -> numpy.typing.NDArray[numpy.float64]:
        """(self: mujoco._structs._MjTimerStatList) -> numpy.typing.NDArray[numpy.float64]"""
    @property
    def number(self) -> numpy.typing.NDArray[numpy.int32]:
        """(self: mujoco._structs._MjTimerStatList) -> numpy.typing.NDArray[numpy.int32]"""

class _MjWarningStatList:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @overload
    def __getitem__(self, arg0: typing.SupportsInt) -> MjWarningStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: typing.SupportsInt) -> mujoco._structs.MjWarningStat

        2. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: mujoco._enums.mjtWarning) -> mujoco._structs.MjWarningStat

        3. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: slice) -> mujoco._structs._MjWarningStatList
        """
    @overload
    def __getitem__(self, arg0: mujoco._enums.mjtWarning) -> MjWarningStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: typing.SupportsInt) -> mujoco._structs.MjWarningStat

        2. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: mujoco._enums.mjtWarning) -> mujoco._structs.MjWarningStat

        3. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: slice) -> mujoco._structs._MjWarningStatList
        """
    @overload
    def __getitem__(self, arg0: slice) -> _MjWarningStatList:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: typing.SupportsInt) -> mujoco._structs.MjWarningStat

        2. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: mujoco._enums.mjtWarning) -> mujoco._structs.MjWarningStat

        3. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: slice) -> mujoco._structs._MjWarningStatList
        """
    def __len__(self) -> int:
        """__len__(self: mujoco._structs._MjWarningStatList) -> int"""
    @property
    def lastinfo(self) -> numpy.typing.NDArray[numpy.int32]:
        """(self: mujoco._structs._MjWarningStatList) -> numpy.typing.NDArray[numpy.int32]"""
    @property
    def number(self) -> numpy.typing.NDArray[numpy.int32]:
        """(self: mujoco._structs._MjWarningStatList) -> numpy.typing.NDArray[numpy.int32]"""

def _recompile_spec_addr(arg0: typing.SupportsInt, arg1: MjModel, arg2: MjData) -> tuple:
    """_recompile_spec_addr(arg0: typing.SupportsInt, arg1: mujoco._structs.MjModel, arg2: mujoco._structs.MjData) -> tuple"""
def mjv_averageCamera(cam1: MjvGLCamera, cam2: MjvGLCamera) -> MjvGLCamera:
    """mjv_averageCamera(cam1: mujoco._structs.MjvGLCamera, cam2: mujoco._structs.MjvGLCamera) -> mujoco._structs.MjvGLCamera

    Return the average of two OpenGL cameras.
    """
