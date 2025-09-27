import mujoco._enums
import numpy
import typing
from typing import Callable, ClassVar, overload

class MjContact:
    H: numpy.ndarray[numpy.float64]
    dim: int
    dist: float
    efc_address: int
    elem: numpy.ndarray[numpy.int32]
    exclude: int
    flex: numpy.ndarray[numpy.int32]
    frame: numpy.ndarray[numpy.float64]
    friction: numpy.ndarray[numpy.float64]
    geom: numpy.ndarray[numpy.int32]
    geom1: int
    geom2: int
    includemargin: float
    mu: float
    pos: numpy.ndarray[numpy.float64]
    solimp: numpy.ndarray[numpy.float64]
    solref: numpy.ndarray[numpy.float64]
    solreffriction: numpy.ndarray[numpy.float64]
    vert: numpy.ndarray[numpy.int32]
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
    B_colind: numpy.ndarray[numpy.int32]
    B_rowadr: numpy.ndarray[numpy.int32]
    B_rownnz: numpy.ndarray[numpy.int32]
    C_colind: numpy.ndarray[numpy.int32]
    C_rowadr: numpy.ndarray[numpy.int32]
    C_rownnz: numpy.ndarray[numpy.int32]
    D_colind: numpy.ndarray[numpy.int32]
    D_diag: numpy.ndarray[numpy.int32]
    D_rowadr: numpy.ndarray[numpy.int32]
    D_rownnz: numpy.ndarray[numpy.int32]
    M_colind: numpy.ndarray[numpy.int32]
    M_rowadr: numpy.ndarray[numpy.int32]
    M_rownnz: numpy.ndarray[numpy.int32]
    act: numpy.ndarray[numpy.float64]
    act_dot: numpy.ndarray[numpy.float64]
    actuator_force: numpy.ndarray[numpy.float64]
    actuator_length: numpy.ndarray[numpy.float64]
    actuator_moment: numpy.ndarray[numpy.float64]
    actuator_velocity: numpy.ndarray[numpy.float64]
    bvh_aabb_dyn: numpy.ndarray[numpy.float64]
    bvh_active: numpy.ndarray[numpy.uint8]
    cacc: numpy.ndarray[numpy.float64]
    cam_xmat: numpy.ndarray[numpy.float64]
    cam_xpos: numpy.ndarray[numpy.float64]
    cdof: numpy.ndarray[numpy.float64]
    cdof_dot: numpy.ndarray[numpy.float64]
    cfrc_ext: numpy.ndarray[numpy.float64]
    cfrc_int: numpy.ndarray[numpy.float64]
    cinert: numpy.ndarray[numpy.float64]
    crb: numpy.ndarray[numpy.float64]
    ctrl: numpy.ndarray[numpy.float64]
    cvel: numpy.ndarray[numpy.float64]
    energy: numpy.ndarray[numpy.float64]
    eq_active: numpy.ndarray[numpy.uint8]
    flexedge_J: numpy.ndarray[numpy.float64]
    flexedge_J_colind: numpy.ndarray[numpy.int32]
    flexedge_J_rowadr: numpy.ndarray[numpy.int32]
    flexedge_J_rownnz: numpy.ndarray[numpy.int32]
    flexedge_length: numpy.ndarray[numpy.float64]
    flexedge_velocity: numpy.ndarray[numpy.float64]
    flexelem_aabb: numpy.ndarray[numpy.float64]
    flexvert_xpos: numpy.ndarray[numpy.float64]
    geom_xmat: numpy.ndarray[numpy.float64]
    geom_xpos: numpy.ndarray[numpy.float64]
    light_xdir: numpy.ndarray[numpy.float64]
    light_xpos: numpy.ndarray[numpy.float64]
    mapD2M: numpy.ndarray[numpy.int32]
    mapM2C: numpy.ndarray[numpy.int32]
    mapM2D: numpy.ndarray[numpy.int32]
    mapM2M: numpy.ndarray[numpy.int32]
    maxuse_arena: int
    maxuse_con: int
    maxuse_efc: int
    maxuse_stack: int
    maxuse_threadstack: numpy.ndarray[numpy.uint64]
    mocap_pos: numpy.ndarray[numpy.float64]
    mocap_quat: numpy.ndarray[numpy.float64]
    moment_colind: numpy.ndarray[numpy.int32]
    moment_rowadr: numpy.ndarray[numpy.int32]
    moment_rownnz: numpy.ndarray[numpy.int32]
    nA: int
    nJ: int
    narena: int
    nbuffer: int
    ncon: int
    ne: int
    nefc: int
    nf: int
    nisland: int
    nl: int
    nplugin: int
    parena: int
    pbase: int
    plugin: numpy.ndarray[numpy.int32]
    plugin_data: numpy.ndarray[numpy.uint64]
    plugin_state: numpy.ndarray[numpy.float64]
    pstack: int
    qDeriv: numpy.ndarray[numpy.float64]
    qH: numpy.ndarray[numpy.float64]
    qHDiagInv: numpy.ndarray[numpy.float64]
    qLD: numpy.ndarray[numpy.float64]
    qLDiagInv: numpy.ndarray[numpy.float64]
    qLU: numpy.ndarray[numpy.float64]
    qM: numpy.ndarray[numpy.float64]
    qacc: numpy.ndarray[numpy.float64]
    qacc_smooth: numpy.ndarray[numpy.float64]
    qacc_warmstart: numpy.ndarray[numpy.float64]
    qfrc_actuator: numpy.ndarray[numpy.float64]
    qfrc_applied: numpy.ndarray[numpy.float64]
    qfrc_bias: numpy.ndarray[numpy.float64]
    qfrc_constraint: numpy.ndarray[numpy.float64]
    qfrc_damper: numpy.ndarray[numpy.float64]
    qfrc_fluid: numpy.ndarray[numpy.float64]
    qfrc_gravcomp: numpy.ndarray[numpy.float64]
    qfrc_inverse: numpy.ndarray[numpy.float64]
    qfrc_passive: numpy.ndarray[numpy.float64]
    qfrc_smooth: numpy.ndarray[numpy.float64]
    qfrc_spring: numpy.ndarray[numpy.float64]
    qpos: numpy.ndarray[numpy.float64]
    qvel: numpy.ndarray[numpy.float64]
    sensordata: numpy.ndarray[numpy.float64]
    site_xmat: numpy.ndarray[numpy.float64]
    site_xpos: numpy.ndarray[numpy.float64]
    solver_fwdinv: numpy.ndarray[numpy.float64]
    solver_niter: numpy.ndarray[numpy.int32]
    solver_nnz: numpy.ndarray[numpy.int32]
    subtree_angmom: numpy.ndarray[numpy.float64]
    subtree_com: numpy.ndarray[numpy.float64]
    subtree_linvel: numpy.ndarray[numpy.float64]
    ten_J: numpy.ndarray[numpy.float64]
    ten_J_colind: numpy.ndarray[numpy.int32]
    ten_J_rowadr: numpy.ndarray[numpy.int32]
    ten_J_rownnz: numpy.ndarray[numpy.int32]
    ten_length: numpy.ndarray[numpy.float64]
    ten_velocity: numpy.ndarray[numpy.float64]
    ten_wrapadr: numpy.ndarray[numpy.int32]
    ten_wrapnum: numpy.ndarray[numpy.int32]
    threadpool: int
    time: float
    userdata: numpy.ndarray[numpy.float64]
    wrap_obj: numpy.ndarray[numpy.int32]
    wrap_xpos: numpy.ndarray[numpy.float64]
    xanchor: numpy.ndarray[numpy.float64]
    xaxis: numpy.ndarray[numpy.float64]
    xfrc_applied: numpy.ndarray[numpy.float64]
    ximat: numpy.ndarray[numpy.float64]
    xipos: numpy.ndarray[numpy.float64]
    xmat: numpy.ndarray[numpy.float64]
    xpos: numpy.ndarray[numpy.float64]
    xquat: numpy.ndarray[numpy.float64]
    def __init__(self, arg0: MjModel) -> None:
        """__init__(self: mujoco._structs.MjData, arg0: mujoco._structs.MjModel) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def actuator(self, *args, **kwargs):
        """actuator(*args, **kwargs)
        Overloaded function.

        1. actuator(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataActuatorViews

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

        1. body(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataBodyViews

        2. body(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataBodyViews
        """
    def cam(self, *args, **kwargs):
        """cam(*args, **kwargs)
        Overloaded function.

        1. cam(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataCameraViews

        2. cam(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataCameraViews
        """
    def camera(self, *args, **kwargs):
        """camera(*args, **kwargs)
        Overloaded function.

        1. camera(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataCameraViews

        2. camera(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataCameraViews
        """
    def geom(self, *args, **kwargs):
        """geom(*args, **kwargs)
        Overloaded function.

        1. geom(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataGeomViews

        2. geom(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataGeomViews
        """
    def jnt(self, *args, **kwargs):
        """jnt(*args, **kwargs)
        Overloaded function.

        1. jnt(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataJointViews

        2. jnt(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataJointViews
        """
    def joint(self, *args, **kwargs):
        """joint(*args, **kwargs)
        Overloaded function.

        1. joint(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataJointViews

        2. joint(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataJointViews
        """
    def light(self, *args, **kwargs):
        """light(*args, **kwargs)
        Overloaded function.

        1. light(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataLightViews

        2. light(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataLightViews
        """
    def sensor(self, *args, **kwargs):
        """sensor(*args, **kwargs)
        Overloaded function.

        1. sensor(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataSensorViews

        2. sensor(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataSensorViews
        """
    def site(self, *args, **kwargs):
        """site(*args, **kwargs)
        Overloaded function.

        1. site(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataSiteViews

        2. site(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataSiteViews
        """
    def ten(self, *args, **kwargs):
        """ten(*args, **kwargs)
        Overloaded function.

        1. ten(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataTendonViews

        2. ten(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataTendonViews
        """
    def tendon(self, *args, **kwargs):
        """tendon(*args, **kwargs)
        Overloaded function.

        1. tendon(self: mujoco._structs.MjData, arg0: int) -> mujoco::python::MjDataTendonViews

        2. tendon(self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataTendonViews
        """
    def __copy__(self) -> MjData:
        """__copy__(self: mujoco._structs.MjData) -> mujoco._structs.MjData"""
    def __deepcopy__(self, arg0: dict) -> MjData:
        """__deepcopy__(self: mujoco._structs.MjData, arg0: dict) -> mujoco._structs.MjData"""
    @property
    def _address(self) -> int: ...
    @property
    def contact(self) -> _MjContactList: ...
    @property
    def dof_island(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def dof_islandind(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_AR(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_AR_colind(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_AR_rowadr(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_AR_rownnz(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_D(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_J(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_JT(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_JT_colind(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_JT_rowadr(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_JT_rownnz(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_JT_rowsuper(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_J_colind(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_J_rowadr(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_J_rownnz(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_J_rowsuper(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_KBIP(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_R(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_aref(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_b(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_diagApprox(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_force(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_frictionloss(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_id(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_island(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_margin(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_pos(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_state(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_type(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def efc_vel(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def island_dofadr(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def island_dofind(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def island_dofnum(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def island_efcadr(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def island_efcind(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def island_efcnum(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def model(self) -> MjModel: ...
    @property
    def signature(self) -> int: ...
    @property
    def solver(self) -> _MjSolverStatList: ...
    @property
    def tendon_efcadr(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def timer(self) -> _MjTimerStatList: ...
    @property
    def warning(self) -> _MjWarningStatList: ...

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
    actuator_acc0: numpy.ndarray[numpy.float64]
    actuator_actadr: numpy.ndarray[numpy.int32]
    actuator_actearly: numpy.ndarray[numpy.uint8]
    actuator_actlimited: numpy.ndarray[numpy.uint8]
    actuator_actnum: numpy.ndarray[numpy.int32]
    actuator_actrange: numpy.ndarray[numpy.float64]
    actuator_biasprm: numpy.ndarray[numpy.float64]
    actuator_biastype: numpy.ndarray[numpy.int32]
    actuator_cranklength: numpy.ndarray[numpy.float64]
    actuator_ctrllimited: numpy.ndarray[numpy.uint8]
    actuator_ctrlrange: numpy.ndarray[numpy.float64]
    actuator_dynprm: numpy.ndarray[numpy.float64]
    actuator_dyntype: numpy.ndarray[numpy.int32]
    actuator_forcelimited: numpy.ndarray[numpy.uint8]
    actuator_forcerange: numpy.ndarray[numpy.float64]
    actuator_gainprm: numpy.ndarray[numpy.float64]
    actuator_gaintype: numpy.ndarray[numpy.int32]
    actuator_gear: numpy.ndarray[numpy.float64]
    actuator_group: numpy.ndarray[numpy.int32]
    actuator_length0: numpy.ndarray[numpy.float64]
    actuator_lengthrange: numpy.ndarray[numpy.float64]
    actuator_plugin: numpy.ndarray[numpy.int32]
    actuator_trnid: numpy.ndarray[numpy.int32]
    actuator_trntype: numpy.ndarray[numpy.int32]
    actuator_user: numpy.ndarray[numpy.float64]
    body_bvhadr: numpy.ndarray[numpy.int32]
    body_bvhnum: numpy.ndarray[numpy.int32]
    body_conaffinity: numpy.ndarray[numpy.int32]
    body_contype: numpy.ndarray[numpy.int32]
    body_dofadr: numpy.ndarray[numpy.int32]
    body_dofnum: numpy.ndarray[numpy.int32]
    body_geomadr: numpy.ndarray[numpy.int32]
    body_geomnum: numpy.ndarray[numpy.int32]
    body_gravcomp: numpy.ndarray[numpy.float64]
    body_inertia: numpy.ndarray[numpy.float64]
    body_invweight0: numpy.ndarray[numpy.float64]
    body_ipos: numpy.ndarray[numpy.float64]
    body_iquat: numpy.ndarray[numpy.float64]
    body_jntadr: numpy.ndarray[numpy.int32]
    body_jntnum: numpy.ndarray[numpy.int32]
    body_margin: numpy.ndarray[numpy.float64]
    body_mass: numpy.ndarray[numpy.float64]
    body_mocapid: numpy.ndarray[numpy.int32]
    body_parentid: numpy.ndarray[numpy.int32]
    body_plugin: numpy.ndarray[numpy.int32]
    body_pos: numpy.ndarray[numpy.float64]
    body_quat: numpy.ndarray[numpy.float64]
    body_rootid: numpy.ndarray[numpy.int32]
    body_sameframe: numpy.ndarray[numpy.uint8]
    body_simple: numpy.ndarray[numpy.uint8]
    body_subtreemass: numpy.ndarray[numpy.float64]
    body_treeid: numpy.ndarray[numpy.int32]
    body_user: numpy.ndarray[numpy.float64]
    body_weldid: numpy.ndarray[numpy.int32]
    bvh_aabb: numpy.ndarray[numpy.float64]
    bvh_child: numpy.ndarray[numpy.int32]
    bvh_depth: numpy.ndarray[numpy.int32]
    bvh_nodeid: numpy.ndarray[numpy.int32]
    cam_bodyid: numpy.ndarray[numpy.int32]
    cam_fovy: numpy.ndarray[numpy.float64]
    cam_intrinsic: numpy.ndarray[numpy.float32]
    cam_ipd: numpy.ndarray[numpy.float64]
    cam_mat0: numpy.ndarray[numpy.float64]
    cam_mode: numpy.ndarray[numpy.int32]
    cam_orthographic: numpy.ndarray[numpy.int32]
    cam_pos: numpy.ndarray[numpy.float64]
    cam_pos0: numpy.ndarray[numpy.float64]
    cam_poscom0: numpy.ndarray[numpy.float64]
    cam_quat: numpy.ndarray[numpy.float64]
    cam_resolution: numpy.ndarray[numpy.int32]
    cam_sensorsize: numpy.ndarray[numpy.float32]
    cam_targetbodyid: numpy.ndarray[numpy.int32]
    cam_user: numpy.ndarray[numpy.float64]
    dof_M0: numpy.ndarray[numpy.float64]
    dof_Madr: numpy.ndarray[numpy.int32]
    dof_armature: numpy.ndarray[numpy.float64]
    dof_bodyid: numpy.ndarray[numpy.int32]
    dof_damping: numpy.ndarray[numpy.float64]
    dof_frictionloss: numpy.ndarray[numpy.float64]
    dof_invweight0: numpy.ndarray[numpy.float64]
    dof_jntid: numpy.ndarray[numpy.int32]
    dof_parentid: numpy.ndarray[numpy.int32]
    dof_simplenum: numpy.ndarray[numpy.int32]
    dof_solimp: numpy.ndarray[numpy.float64]
    dof_solref: numpy.ndarray[numpy.float64]
    dof_treeid: numpy.ndarray[numpy.int32]
    eq_active0: numpy.ndarray[numpy.uint8]
    eq_data: numpy.ndarray[numpy.float64]
    eq_obj1id: numpy.ndarray[numpy.int32]
    eq_obj2id: numpy.ndarray[numpy.int32]
    eq_objtype: numpy.ndarray[numpy.int32]
    eq_solimp: numpy.ndarray[numpy.float64]
    eq_solref: numpy.ndarray[numpy.float64]
    eq_type: numpy.ndarray[numpy.int32]
    exclude_signature: numpy.ndarray[numpy.int32]
    flex_activelayers: numpy.ndarray[numpy.int32]
    flex_bvhadr: numpy.ndarray[numpy.int32]
    flex_bvhnum: numpy.ndarray[numpy.int32]
    flex_centered: numpy.ndarray[numpy.uint8]
    flex_conaffinity: numpy.ndarray[numpy.int32]
    flex_condim: numpy.ndarray[numpy.int32]
    flex_contype: numpy.ndarray[numpy.int32]
    flex_damping: numpy.ndarray[numpy.float64]
    flex_dim: numpy.ndarray[numpy.int32]
    flex_edge: numpy.ndarray[numpy.int32]
    flex_edgeadr: numpy.ndarray[numpy.int32]
    flex_edgedamping: numpy.ndarray[numpy.float64]
    flex_edgeequality: numpy.ndarray[numpy.uint8]
    flex_edgenum: numpy.ndarray[numpy.int32]
    flex_edgestiffness: numpy.ndarray[numpy.float64]
    flex_elem: numpy.ndarray[numpy.int32]
    flex_elemadr: numpy.ndarray[numpy.int32]
    flex_elemdataadr: numpy.ndarray[numpy.int32]
    flex_elemedge: numpy.ndarray[numpy.int32]
    flex_elemedgeadr: numpy.ndarray[numpy.int32]
    flex_elemlayer: numpy.ndarray[numpy.int32]
    flex_elemnum: numpy.ndarray[numpy.int32]
    flex_elemtexcoord: numpy.ndarray[numpy.int32]
    flex_evpair: numpy.ndarray[numpy.int32]
    flex_evpairadr: numpy.ndarray[numpy.int32]
    flex_evpairnum: numpy.ndarray[numpy.int32]
    flex_flatskin: numpy.ndarray[numpy.uint8]
    flex_friction: numpy.ndarray[numpy.float64]
    flex_gap: numpy.ndarray[numpy.float64]
    flex_group: numpy.ndarray[numpy.int32]
    flex_internal: numpy.ndarray[numpy.uint8]
    flex_interp: numpy.ndarray[numpy.int32]
    flex_margin: numpy.ndarray[numpy.float64]
    flex_matid: numpy.ndarray[numpy.int32]
    flex_node: numpy.ndarray[numpy.float64]
    flex_node0: numpy.ndarray[numpy.float64]
    flex_nodeadr: numpy.ndarray[numpy.int32]
    flex_nodebodyid: numpy.ndarray[numpy.int32]
    flex_nodenum: numpy.ndarray[numpy.int32]
    flex_priority: numpy.ndarray[numpy.int32]
    flex_radius: numpy.ndarray[numpy.float64]
    flex_rgba: numpy.ndarray[numpy.float32]
    flex_rigid: numpy.ndarray[numpy.uint8]
    flex_selfcollide: numpy.ndarray[numpy.int32]
    flex_shell: numpy.ndarray[numpy.int32]
    flex_shelldataadr: numpy.ndarray[numpy.int32]
    flex_shellnum: numpy.ndarray[numpy.int32]
    flex_solimp: numpy.ndarray[numpy.float64]
    flex_solmix: numpy.ndarray[numpy.float64]
    flex_solref: numpy.ndarray[numpy.float64]
    flex_stiffness: numpy.ndarray[numpy.float64]
    flex_texcoord: numpy.ndarray[numpy.float32]
    flex_texcoordadr: numpy.ndarray[numpy.int32]
    flex_vert: numpy.ndarray[numpy.float64]
    flex_vert0: numpy.ndarray[numpy.float64]
    flex_vertadr: numpy.ndarray[numpy.int32]
    flex_vertbodyid: numpy.ndarray[numpy.int32]
    flex_vertnum: numpy.ndarray[numpy.int32]
    flexedge_invweight0: numpy.ndarray[numpy.float64]
    flexedge_length0: numpy.ndarray[numpy.float64]
    flexedge_rigid: numpy.ndarray[numpy.uint8]
    geom_aabb: numpy.ndarray[numpy.float64]
    geom_bodyid: numpy.ndarray[numpy.int32]
    geom_conaffinity: numpy.ndarray[numpy.int32]
    geom_condim: numpy.ndarray[numpy.int32]
    geom_contype: numpy.ndarray[numpy.int32]
    geom_dataid: numpy.ndarray[numpy.int32]
    geom_fluid: numpy.ndarray[numpy.float64]
    geom_friction: numpy.ndarray[numpy.float64]
    geom_gap: numpy.ndarray[numpy.float64]
    geom_group: numpy.ndarray[numpy.int32]
    geom_margin: numpy.ndarray[numpy.float64]
    geom_matid: numpy.ndarray[numpy.int32]
    geom_plugin: numpy.ndarray[numpy.int32]
    geom_pos: numpy.ndarray[numpy.float64]
    geom_priority: numpy.ndarray[numpy.int32]
    geom_quat: numpy.ndarray[numpy.float64]
    geom_rbound: numpy.ndarray[numpy.float64]
    geom_rgba: numpy.ndarray[numpy.float32]
    geom_sameframe: numpy.ndarray[numpy.uint8]
    geom_size: numpy.ndarray[numpy.float64]
    geom_solimp: numpy.ndarray[numpy.float64]
    geom_solmix: numpy.ndarray[numpy.float64]
    geom_solref: numpy.ndarray[numpy.float64]
    geom_type: numpy.ndarray[numpy.int32]
    geom_user: numpy.ndarray[numpy.float64]
    hfield_adr: numpy.ndarray[numpy.int32]
    hfield_data: numpy.ndarray[numpy.float32]
    hfield_ncol: numpy.ndarray[numpy.int32]
    hfield_nrow: numpy.ndarray[numpy.int32]
    hfield_pathadr: numpy.ndarray[numpy.int32]
    hfield_size: numpy.ndarray[numpy.float64]
    jnt_actfrclimited: numpy.ndarray[numpy.uint8]
    jnt_actfrcrange: numpy.ndarray[numpy.float64]
    jnt_actgravcomp: numpy.ndarray[numpy.uint8]
    jnt_axis: numpy.ndarray[numpy.float64]
    jnt_bodyid: numpy.ndarray[numpy.int32]
    jnt_dofadr: numpy.ndarray[numpy.int32]
    jnt_group: numpy.ndarray[numpy.int32]
    jnt_limited: numpy.ndarray[numpy.uint8]
    jnt_margin: numpy.ndarray[numpy.float64]
    jnt_pos: numpy.ndarray[numpy.float64]
    jnt_qposadr: numpy.ndarray[numpy.int32]
    jnt_range: numpy.ndarray[numpy.float64]
    jnt_solimp: numpy.ndarray[numpy.float64]
    jnt_solref: numpy.ndarray[numpy.float64]
    jnt_stiffness: numpy.ndarray[numpy.float64]
    jnt_type: numpy.ndarray[numpy.int32]
    jnt_user: numpy.ndarray[numpy.float64]
    key_act: numpy.ndarray[numpy.float64]
    key_ctrl: numpy.ndarray[numpy.float64]
    key_mpos: numpy.ndarray[numpy.float64]
    key_mquat: numpy.ndarray[numpy.float64]
    key_qpos: numpy.ndarray[numpy.float64]
    key_qvel: numpy.ndarray[numpy.float64]
    key_time: numpy.ndarray[numpy.float64]
    light_active: numpy.ndarray[numpy.uint8]
    light_ambient: numpy.ndarray[numpy.float32]
    light_attenuation: numpy.ndarray[numpy.float32]
    light_bodyid: numpy.ndarray[numpy.int32]
    light_bulbradius: numpy.ndarray[numpy.float32]
    light_castshadow: numpy.ndarray[numpy.uint8]
    light_cutoff: numpy.ndarray[numpy.float32]
    light_diffuse: numpy.ndarray[numpy.float32]
    light_dir: numpy.ndarray[numpy.float64]
    light_dir0: numpy.ndarray[numpy.float64]
    light_directional: numpy.ndarray[numpy.uint8]
    light_exponent: numpy.ndarray[numpy.float32]
    light_mode: numpy.ndarray[numpy.int32]
    light_pos: numpy.ndarray[numpy.float64]
    light_pos0: numpy.ndarray[numpy.float64]
    light_poscom0: numpy.ndarray[numpy.float64]
    light_specular: numpy.ndarray[numpy.float32]
    light_targetbodyid: numpy.ndarray[numpy.int32]
    mat_emission: numpy.ndarray[numpy.float32]
    mat_metallic: numpy.ndarray[numpy.float32]
    mat_reflectance: numpy.ndarray[numpy.float32]
    mat_rgba: numpy.ndarray[numpy.float32]
    mat_roughness: numpy.ndarray[numpy.float32]
    mat_shininess: numpy.ndarray[numpy.float32]
    mat_specular: numpy.ndarray[numpy.float32]
    mat_texid: numpy.ndarray[numpy.int32]
    mat_texrepeat: numpy.ndarray[numpy.float32]
    mat_texuniform: numpy.ndarray[numpy.uint8]
    mesh_bvhadr: numpy.ndarray[numpy.int32]
    mesh_bvhnum: numpy.ndarray[numpy.int32]
    mesh_face: numpy.ndarray[numpy.int32]
    mesh_faceadr: numpy.ndarray[numpy.int32]
    mesh_facenormal: numpy.ndarray[numpy.int32]
    mesh_facenum: numpy.ndarray[numpy.int32]
    mesh_facetexcoord: numpy.ndarray[numpy.int32]
    mesh_graph: numpy.ndarray[numpy.int32]
    mesh_graphadr: numpy.ndarray[numpy.int32]
    mesh_normal: numpy.ndarray[numpy.float32]
    mesh_normaladr: numpy.ndarray[numpy.int32]
    mesh_normalnum: numpy.ndarray[numpy.int32]
    mesh_pathadr: numpy.ndarray[numpy.int32]
    mesh_polyadr: numpy.ndarray[numpy.int32]
    mesh_polymap: numpy.ndarray[numpy.int32]
    mesh_polymapadr: numpy.ndarray[numpy.int32]
    mesh_polymapnum: numpy.ndarray[numpy.int32]
    mesh_polynormal: numpy.ndarray[numpy.float64]
    mesh_polynum: numpy.ndarray[numpy.int32]
    mesh_polyvert: numpy.ndarray[numpy.int32]
    mesh_polyvertadr: numpy.ndarray[numpy.int32]
    mesh_polyvertnum: numpy.ndarray[numpy.int32]
    mesh_pos: numpy.ndarray[numpy.float64]
    mesh_quat: numpy.ndarray[numpy.float64]
    mesh_scale: numpy.ndarray[numpy.float64]
    mesh_texcoord: numpy.ndarray[numpy.float32]
    mesh_texcoordadr: numpy.ndarray[numpy.int32]
    mesh_texcoordnum: numpy.ndarray[numpy.int32]
    mesh_vert: numpy.ndarray[numpy.float32]
    mesh_vertadr: numpy.ndarray[numpy.int32]
    mesh_vertnum: numpy.ndarray[numpy.int32]
    name_actuatoradr: numpy.ndarray[numpy.int32]
    name_bodyadr: numpy.ndarray[numpy.int32]
    name_camadr: numpy.ndarray[numpy.int32]
    name_eqadr: numpy.ndarray[numpy.int32]
    name_excludeadr: numpy.ndarray[numpy.int32]
    name_flexadr: numpy.ndarray[numpy.int32]
    name_geomadr: numpy.ndarray[numpy.int32]
    name_hfieldadr: numpy.ndarray[numpy.int32]
    name_jntadr: numpy.ndarray[numpy.int32]
    name_keyadr: numpy.ndarray[numpy.int32]
    name_lightadr: numpy.ndarray[numpy.int32]
    name_matadr: numpy.ndarray[numpy.int32]
    name_meshadr: numpy.ndarray[numpy.int32]
    name_numericadr: numpy.ndarray[numpy.int32]
    name_pairadr: numpy.ndarray[numpy.int32]
    name_pluginadr: numpy.ndarray[numpy.int32]
    name_sensoradr: numpy.ndarray[numpy.int32]
    name_siteadr: numpy.ndarray[numpy.int32]
    name_skinadr: numpy.ndarray[numpy.int32]
    name_tendonadr: numpy.ndarray[numpy.int32]
    name_texadr: numpy.ndarray[numpy.int32]
    name_textadr: numpy.ndarray[numpy.int32]
    name_tupleadr: numpy.ndarray[numpy.int32]
    names_map: numpy.ndarray[numpy.int32]
    numeric_adr: numpy.ndarray[numpy.int32]
    numeric_data: numpy.ndarray[numpy.float64]
    numeric_size: numpy.ndarray[numpy.int32]
    pair_dim: numpy.ndarray[numpy.int32]
    pair_friction: numpy.ndarray[numpy.float64]
    pair_gap: numpy.ndarray[numpy.float64]
    pair_geom1: numpy.ndarray[numpy.int32]
    pair_geom2: numpy.ndarray[numpy.int32]
    pair_margin: numpy.ndarray[numpy.float64]
    pair_signature: numpy.ndarray[numpy.int32]
    pair_solimp: numpy.ndarray[numpy.float64]
    pair_solref: numpy.ndarray[numpy.float64]
    pair_solreffriction: numpy.ndarray[numpy.float64]
    plugin: numpy.ndarray[numpy.int32]
    plugin_attr: numpy.ndarray[numpy.int8]
    plugin_attradr: numpy.ndarray[numpy.int32]
    plugin_stateadr: numpy.ndarray[numpy.int32]
    plugin_statenum: numpy.ndarray[numpy.int32]
    qpos0: numpy.ndarray[numpy.float64]
    qpos_spring: numpy.ndarray[numpy.float64]
    sensor_adr: numpy.ndarray[numpy.int32]
    sensor_cutoff: numpy.ndarray[numpy.float64]
    sensor_datatype: numpy.ndarray[numpy.int32]
    sensor_dim: numpy.ndarray[numpy.int32]
    sensor_needstage: numpy.ndarray[numpy.int32]
    sensor_noise: numpy.ndarray[numpy.float64]
    sensor_objid: numpy.ndarray[numpy.int32]
    sensor_objtype: numpy.ndarray[numpy.int32]
    sensor_plugin: numpy.ndarray[numpy.int32]
    sensor_refid: numpy.ndarray[numpy.int32]
    sensor_reftype: numpy.ndarray[numpy.int32]
    sensor_type: numpy.ndarray[numpy.int32]
    sensor_user: numpy.ndarray[numpy.float64]
    site_bodyid: numpy.ndarray[numpy.int32]
    site_group: numpy.ndarray[numpy.int32]
    site_matid: numpy.ndarray[numpy.int32]
    site_pos: numpy.ndarray[numpy.float64]
    site_quat: numpy.ndarray[numpy.float64]
    site_rgba: numpy.ndarray[numpy.float32]
    site_sameframe: numpy.ndarray[numpy.uint8]
    site_size: numpy.ndarray[numpy.float64]
    site_type: numpy.ndarray[numpy.int32]
    site_user: numpy.ndarray[numpy.float64]
    skin_boneadr: numpy.ndarray[numpy.int32]
    skin_bonebindpos: numpy.ndarray[numpy.float32]
    skin_bonebindquat: numpy.ndarray[numpy.float32]
    skin_bonebodyid: numpy.ndarray[numpy.int32]
    skin_bonenum: numpy.ndarray[numpy.int32]
    skin_bonevertadr: numpy.ndarray[numpy.int32]
    skin_bonevertid: numpy.ndarray[numpy.int32]
    skin_bonevertnum: numpy.ndarray[numpy.int32]
    skin_bonevertweight: numpy.ndarray[numpy.float32]
    skin_face: numpy.ndarray[numpy.int32]
    skin_faceadr: numpy.ndarray[numpy.int32]
    skin_facenum: numpy.ndarray[numpy.int32]
    skin_group: numpy.ndarray[numpy.int32]
    skin_inflate: numpy.ndarray[numpy.float32]
    skin_matid: numpy.ndarray[numpy.int32]
    skin_pathadr: numpy.ndarray[numpy.int32]
    skin_rgba: numpy.ndarray[numpy.float32]
    skin_texcoord: numpy.ndarray[numpy.float32]
    skin_texcoordadr: numpy.ndarray[numpy.int32]
    skin_vert: numpy.ndarray[numpy.float32]
    skin_vertadr: numpy.ndarray[numpy.int32]
    skin_vertnum: numpy.ndarray[numpy.int32]
    tendon_actfrclimited: numpy.ndarray[numpy.uint8]
    tendon_actfrcrange: numpy.ndarray[numpy.float64]
    tendon_adr: numpy.ndarray[numpy.int32]
    tendon_armature: numpy.ndarray[numpy.float64]
    tendon_damping: numpy.ndarray[numpy.float64]
    tendon_frictionloss: numpy.ndarray[numpy.float64]
    tendon_group: numpy.ndarray[numpy.int32]
    tendon_invweight0: numpy.ndarray[numpy.float64]
    tendon_length0: numpy.ndarray[numpy.float64]
    tendon_lengthspring: numpy.ndarray[numpy.float64]
    tendon_limited: numpy.ndarray[numpy.uint8]
    tendon_margin: numpy.ndarray[numpy.float64]
    tendon_matid: numpy.ndarray[numpy.int32]
    tendon_num: numpy.ndarray[numpy.int32]
    tendon_range: numpy.ndarray[numpy.float64]
    tendon_rgba: numpy.ndarray[numpy.float32]
    tendon_solimp_fri: numpy.ndarray[numpy.float64]
    tendon_solimp_lim: numpy.ndarray[numpy.float64]
    tendon_solref_fri: numpy.ndarray[numpy.float64]
    tendon_solref_lim: numpy.ndarray[numpy.float64]
    tendon_stiffness: numpy.ndarray[numpy.float64]
    tendon_user: numpy.ndarray[numpy.float64]
    tendon_width: numpy.ndarray[numpy.float64]
    tex_adr: numpy.ndarray[numpy.int32]
    tex_data: numpy.ndarray[numpy.uint8]
    tex_height: numpy.ndarray[numpy.int32]
    tex_nchannel: numpy.ndarray[numpy.int32]
    tex_pathadr: numpy.ndarray[numpy.int32]
    tex_type: numpy.ndarray[numpy.int32]
    tex_width: numpy.ndarray[numpy.int32]
    text_adr: numpy.ndarray[numpy.int32]
    text_size: numpy.ndarray[numpy.int32]
    tuple_adr: numpy.ndarray[numpy.int32]
    tuple_objid: numpy.ndarray[numpy.int32]
    tuple_objprm: numpy.ndarray[numpy.float64]
    tuple_objtype: numpy.ndarray[numpy.int32]
    tuple_size: numpy.ndarray[numpy.int32]
    wrap_objid: numpy.ndarray[numpy.int32]
    wrap_prm: numpy.ndarray[numpy.float64]
    wrap_type: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def _from_model_ptr(arg0: int) -> MjModel:
        """_from_model_ptr(arg0: int) -> mujoco._structs.MjModel"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def actuator(self, *args, **kwargs):
        """actuator(*args, **kwargs)
        Overloaded function.

        1. actuator(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelActuatorViews

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

        1. body(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelBodyViews

        2. body(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelBodyViews
        """
    def cam(self, *args, **kwargs):
        """cam(*args, **kwargs)
        Overloaded function.

        1. cam(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelCameraViews

        2. cam(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelCameraViews
        """
    def camera(self, *args, **kwargs):
        """camera(*args, **kwargs)
        Overloaded function.

        1. camera(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelCameraViews

        2. camera(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelCameraViews
        """
    def eq(self, *args, **kwargs):
        """eq(*args, **kwargs)
        Overloaded function.

        1. eq(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelEqualityViews

        2. eq(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelEqualityViews
        """
    def equality(self, *args, **kwargs):
        """equality(*args, **kwargs)
        Overloaded function.

        1. equality(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelEqualityViews

        2. equality(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelEqualityViews
        """
    def exclude(self, *args, **kwargs):
        """exclude(*args, **kwargs)
        Overloaded function.

        1. exclude(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelExcludeViews

        2. exclude(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelExcludeViews
        """
    @staticmethod
    def from_binary_path(filename: str, assets: dict[str, bytes] | None = ...) -> MjModel:
        """from_binary_path(filename: str, assets: Optional[dict[str, bytes]] = None) -> mujoco._structs.MjModel

        Loads an MjModel from an MJB file and an optional assets dictionary.

        The filename for the MJB can also refer to a key in the assets dictionary.
        This is useful for example when the MJB is not available as a file on disk.
        """
    @staticmethod
    def from_xml_path(filename: str, assets: dict[str, bytes] | None = ...) -> MjModel:
        """from_xml_path(filename: str, assets: Optional[dict[str, bytes]] = None) -> mujoco._structs.MjModel

        Loads an MjModel from an XML file and an optional assets dictionary.

        The filename for the XML can also refer to a key in the assets dictionary.
        This is useful for example when the XML is not available as a file on disk.
        """
    @staticmethod
    def from_xml_string(xml: str, assets: dict[str, bytes] | None = ...) -> MjModel:
        """from_xml_string(xml: str, assets: Optional[dict[str, bytes]] = None) -> mujoco._structs.MjModel

        Loads an MjModel from an XML string and an optional assets dictionary.
        """
    def geom(self, *args, **kwargs):
        """geom(*args, **kwargs)
        Overloaded function.

        1. geom(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelGeomViews

        2. geom(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelGeomViews
        """
    def hfield(self, *args, **kwargs):
        """hfield(*args, **kwargs)
        Overloaded function.

        1. hfield(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelHfieldViews

        2. hfield(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelHfieldViews
        """
    def jnt(self, *args, **kwargs):
        """jnt(*args, **kwargs)
        Overloaded function.

        1. jnt(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelJointViews

        2. jnt(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelJointViews
        """
    def joint(self, *args, **kwargs):
        """joint(*args, **kwargs)
        Overloaded function.

        1. joint(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelJointViews

        2. joint(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelJointViews
        """
    def key(self, *args, **kwargs):
        """key(*args, **kwargs)
        Overloaded function.

        1. key(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelKeyframeViews

        2. key(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelKeyframeViews
        """
    def keyframe(self, *args, **kwargs):
        """keyframe(*args, **kwargs)
        Overloaded function.

        1. keyframe(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelKeyframeViews

        2. keyframe(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelKeyframeViews
        """
    def light(self, *args, **kwargs):
        """light(*args, **kwargs)
        Overloaded function.

        1. light(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelLightViews

        2. light(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelLightViews
        """
    def mat(self, *args, **kwargs):
        """mat(*args, **kwargs)
        Overloaded function.

        1. mat(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelMaterialViews

        2. mat(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelMaterialViews
        """
    def material(self, *args, **kwargs):
        """material(*args, **kwargs)
        Overloaded function.

        1. material(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelMaterialViews

        2. material(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelMaterialViews
        """
    def mesh(self, *args, **kwargs):
        """mesh(*args, **kwargs)
        Overloaded function.

        1. mesh(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelMeshViews

        2. mesh(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelMeshViews
        """
    def numeric(self, *args, **kwargs):
        """numeric(*args, **kwargs)
        Overloaded function.

        1. numeric(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelNumericViews

        2. numeric(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelNumericViews
        """
    def pair(self, *args, **kwargs):
        """pair(*args, **kwargs)
        Overloaded function.

        1. pair(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelPairViews

        2. pair(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelPairViews
        """
    def sensor(self, *args, **kwargs):
        """sensor(*args, **kwargs)
        Overloaded function.

        1. sensor(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelSensorViews

        2. sensor(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelSensorViews
        """
    def site(self, *args, **kwargs):
        """site(*args, **kwargs)
        Overloaded function.

        1. site(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelSiteViews

        2. site(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelSiteViews
        """
    def skin(self, *args, **kwargs):
        """skin(*args, **kwargs)
        Overloaded function.

        1. skin(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelSkinViews

        2. skin(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelSkinViews
        """
    def tendon(self, *args, **kwargs):
        """tendon(*args, **kwargs)
        Overloaded function.

        1. tendon(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelTendonViews

        2. tendon(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelTendonViews
        """
    def tex(self, *args, **kwargs):
        """tex(*args, **kwargs)
        Overloaded function.

        1. tex(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelTextureViews

        2. tex(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelTextureViews
        """
    def texture(self, *args, **kwargs):
        """texture(*args, **kwargs)
        Overloaded function.

        1. texture(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelTextureViews

        2. texture(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelTextureViews
        """
    def tuple(self, *args, **kwargs):
        """tuple(*args, **kwargs)
        Overloaded function.

        1. tuple(self: mujoco._structs.MjModel, arg0: int) -> mujoco::python::MjModelTupleViews

        2. tuple(self: mujoco._structs.MjModel, name: str = '') -> mujoco::python::MjModelTupleViews
        """
    def __copy__(self) -> MjModel:
        """__copy__(self: mujoco._structs.MjModel) -> mujoco._structs.MjModel"""
    def __deepcopy__(self, arg0: dict) -> MjModel:
        """__deepcopy__(self: mujoco._structs.MjModel, arg0: dict) -> mujoco._structs.MjModel"""
    @property
    def _address(self) -> int: ...
    @property
    def _sizes(self) -> numpy.ndarray[numpy.int64]: ...
    @property
    def nB(self) -> int: ...
    @property
    def nC(self) -> int: ...
    @property
    def nD(self) -> int: ...
    @property
    def nJmom(self) -> int: ...
    @property
    def nM(self) -> int: ...
    @property
    def na(self) -> int: ...
    @property
    def names(self) -> bytes: ...
    @property
    def narena(self) -> int: ...
    @property
    def nbody(self) -> int: ...
    @property
    def nbuffer(self) -> int: ...
    @property
    def nbvh(self) -> int: ...
    @property
    def nbvhdynamic(self) -> int: ...
    @property
    def nbvhstatic(self) -> int: ...
    @property
    def ncam(self) -> int: ...
    @property
    def nconmax(self) -> int: ...
    @property
    def nemax(self) -> int: ...
    @property
    def neq(self) -> int: ...
    @property
    def nexclude(self) -> int: ...
    @property
    def nflex(self) -> int: ...
    @property
    def nflexedge(self) -> int: ...
    @property
    def nflexelem(self) -> int: ...
    @property
    def nflexelemdata(self) -> int: ...
    @property
    def nflexelemedge(self) -> int: ...
    @property
    def nflexevpair(self) -> int: ...
    @property
    def nflexnode(self) -> int: ...
    @property
    def nflexshelldata(self) -> int: ...
    @property
    def nflextexcoord(self) -> int: ...
    @property
    def nflexvert(self) -> int: ...
    @property
    def ngeom(self) -> int: ...
    @property
    def ngravcomp(self) -> int: ...
    @property
    def nhfield(self) -> int: ...
    @property
    def nhfielddata(self) -> int: ...
    @property
    def njmax(self) -> int: ...
    @property
    def njnt(self) -> int: ...
    @property
    def nkey(self) -> int: ...
    @property
    def nlight(self) -> int: ...
    @property
    def nmat(self) -> int: ...
    @property
    def nmesh(self) -> int: ...
    @property
    def nmeshface(self) -> int: ...
    @property
    def nmeshgraph(self) -> int: ...
    @property
    def nmeshnormal(self) -> int: ...
    @property
    def nmeshpoly(self) -> int: ...
    @property
    def nmeshpolymap(self) -> int: ...
    @property
    def nmeshpolyvert(self) -> int: ...
    @property
    def nmeshtexcoord(self) -> int: ...
    @property
    def nmeshvert(self) -> int: ...
    @property
    def nmocap(self) -> int: ...
    @property
    def nnames(self) -> int: ...
    @property
    def nnames_map(self) -> int: ...
    @property
    def nnumeric(self) -> int: ...
    @property
    def nnumericdata(self) -> int: ...
    @property
    def npair(self) -> int: ...
    @property
    def npaths(self) -> int: ...
    @property
    def nplugin(self) -> int: ...
    @property
    def npluginattr(self) -> int: ...
    @property
    def npluginstate(self) -> int: ...
    @property
    def nq(self) -> int: ...
    @property
    def nsensor(self) -> int: ...
    @property
    def nsensordata(self) -> int: ...
    @property
    def nsite(self) -> int: ...
    @property
    def nskin(self) -> int: ...
    @property
    def nskinbone(self) -> int: ...
    @property
    def nskinbonevert(self) -> int: ...
    @property
    def nskinface(self) -> int: ...
    @property
    def nskintexvert(self) -> int: ...
    @property
    def nskinvert(self) -> int: ...
    @property
    def ntendon(self) -> int: ...
    @property
    def ntex(self) -> int: ...
    @property
    def ntexdata(self) -> int: ...
    @property
    def ntext(self) -> int: ...
    @property
    def ntextdata(self) -> int: ...
    @property
    def ntree(self) -> int: ...
    @property
    def ntuple(self) -> int: ...
    @property
    def ntupledata(self) -> int: ...
    @property
    def nu(self) -> int: ...
    @property
    def nuser_actuator(self) -> int: ...
    @property
    def nuser_body(self) -> int: ...
    @property
    def nuser_cam(self) -> int: ...
    @property
    def nuser_geom(self) -> int: ...
    @property
    def nuser_jnt(self) -> int: ...
    @property
    def nuser_sensor(self) -> int: ...
    @property
    def nuser_site(self) -> int: ...
    @property
    def nuser_tendon(self) -> int: ...
    @property
    def nuserdata(self) -> int: ...
    @property
    def nv(self) -> int: ...
    @property
    def nwrap(self) -> int: ...
    @property
    def opt(self) -> MjOption: ...
    @property
    def paths(self) -> bytes: ...
    @property
    def signature(self) -> int: ...
    @property
    def stat(self): ...
    @property
    def text_data(self) -> bytes: ...
    @property
    def vis(self) -> MjVisual: ...

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
    gravity: numpy.ndarray[numpy.float64]
    impratio: float
    integrator: int
    iterations: int
    jacobian: int
    ls_iterations: int
    ls_tolerance: float
    magnetic: numpy.ndarray[numpy.float64]
    noslip_iterations: int
    noslip_tolerance: float
    o_friction: numpy.ndarray[numpy.float64]
    o_margin: float
    o_solimp: numpy.ndarray[numpy.float64]
    o_solref: numpy.ndarray[numpy.float64]
    sdf_initpoints: int
    sdf_iterations: int
    solver: int
    timestep: float
    tolerance: float
    viscosity: float
    wind: numpy.ndarray[numpy.float64]
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
    center: numpy.ndarray[numpy.float64]
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
        ambient: numpy.ndarray[numpy.float32]
        diffuse: numpy.ndarray[numpy.float32]
        specular: numpy.ndarray[numpy.float32]
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
        actuator: numpy.ndarray[numpy.float32]
        actuatornegative: numpy.ndarray[numpy.float32]
        actuatorpositive: numpy.ndarray[numpy.float32]
        bv: numpy.ndarray[numpy.float32]
        bvactive: numpy.ndarray[numpy.float32]
        camera: numpy.ndarray[numpy.float32]
        com: numpy.ndarray[numpy.float32]
        connect: numpy.ndarray[numpy.float32]
        constraint: numpy.ndarray[numpy.float32]
        contactforce: numpy.ndarray[numpy.float32]
        contactfriction: numpy.ndarray[numpy.float32]
        contactgap: numpy.ndarray[numpy.float32]
        contactpoint: numpy.ndarray[numpy.float32]
        contacttorque: numpy.ndarray[numpy.float32]
        crankbroken: numpy.ndarray[numpy.float32]
        fog: numpy.ndarray[numpy.float32]
        force: numpy.ndarray[numpy.float32]
        frustum: numpy.ndarray[numpy.float32]
        haze: numpy.ndarray[numpy.float32]
        inertia: numpy.ndarray[numpy.float32]
        joint: numpy.ndarray[numpy.float32]
        light: numpy.ndarray[numpy.float32]
        rangefinder: numpy.ndarray[numpy.float32]
        selectpoint: numpy.ndarray[numpy.float32]
        slidercrank: numpy.ndarray[numpy.float32]
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
    def global_(self) -> MjVisual.Global: ...
    @property
    def headlight(self) -> MjVisual.Headlight: ...
    @property
    def map(self) -> MjVisual.Map: ...
    @property
    def quality(self) -> MjVisual.Quality: ...
    @property
    def rgba(self) -> MjVisual.Rgba: ...
    @property
    def scale(self) -> MjVisual.Scale: ...

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
    lookat: numpy.ndarray[numpy.float64]
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
    figurergba: numpy.ndarray[numpy.float32]
    flg_barplot: int
    flg_extend: int
    flg_legend: int
    flg_selection: int
    flg_symmetric: int
    flg_ticklabel: numpy.ndarray[numpy.int32]
    gridrgb: numpy.ndarray[numpy.float32]
    gridsize: numpy.ndarray[numpy.int32]
    gridwidth: float
    highlight: numpy.ndarray[numpy.int32]
    highlightid: int
    legendoffset: int
    legendrgba: numpy.ndarray[numpy.float32]
    linedata: numpy.ndarray[numpy.float32]
    linepnt: numpy.ndarray[numpy.int32]
    linergb: numpy.ndarray[numpy.float32]
    linewidth: float
    minwidth: str
    panergba: numpy.ndarray[numpy.float32]
    range: numpy.ndarray[numpy.float32]
    selection: float
    subplot: int
    textrgb: numpy.ndarray[numpy.float32]
    title: str
    xaxisdata: numpy.ndarray[numpy.float32]
    xaxispixel: numpy.ndarray[numpy.int32]
    xformat: str
    xlabel: str
    yaxisdata: numpy.ndarray[numpy.float32]
    yaxispixel: numpy.ndarray[numpy.int32]
    yformat: str
    def __init__(self) -> None:
        """__init__(self: mujoco._structs.MjvFigure) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvFigure:
        """__copy__(self: mujoco._structs.MjvFigure) -> mujoco._structs.MjvFigure"""
    def __deepcopy__(self, arg0: dict) -> MjvFigure:
        """__deepcopy__(self: mujoco._structs.MjvFigure, arg0: dict) -> mujoco._structs.MjvFigure"""
    @property
    def linename(self) -> numpy.ndarray: ...

class MjvGLCamera:
    forward: numpy.ndarray[numpy.float32]
    frustum_bottom: float
    frustum_center: float
    frustum_far: float
    frustum_near: float
    frustum_top: float
    frustum_width: float
    orthographic: int
    pos: numpy.ndarray[numpy.float32]
    up: numpy.ndarray[numpy.float32]
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
    mat: numpy.ndarray[numpy.float32]
    matid: int
    modelrbound: float
    objid: int
    objtype: int
    pos: numpy.ndarray[numpy.float32]
    reflectance: float
    rgba: numpy.ndarray[numpy.float32]
    segid: int
    shininess: float
    size: numpy.ndarray[numpy.float32]
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
    ambient: numpy.ndarray[numpy.float32]
    attenuation: numpy.ndarray[numpy.float32]
    bulbradius: float
    castshadow: int
    cutoff: float
    diffuse: numpy.ndarray[numpy.float32]
    dir: numpy.ndarray[numpy.float32]
    directional: int
    exponent: float
    headlight: int
    pos: numpy.ndarray[numpy.float32]
    specular: numpy.ndarray[numpy.float32]
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
    actuatorgroup: numpy.ndarray[numpy.uint8]
    bvh_depth: int
    flags: numpy.ndarray[numpy.uint8]
    flex_layer: int
    flexgroup: numpy.ndarray[numpy.uint8]
    frame: int
    geomgroup: numpy.ndarray[numpy.uint8]
    jointgroup: numpy.ndarray[numpy.uint8]
    label: int
    sitegroup: numpy.ndarray[numpy.uint8]
    skingroup: numpy.ndarray[numpy.uint8]
    tendongroup: numpy.ndarray[numpy.uint8]
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
    localpos: numpy.ndarray[numpy.float64]
    refpos: numpy.ndarray[numpy.float64]
    refquat: numpy.ndarray[numpy.float64]
    refselpos: numpy.ndarray[numpy.float64]
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
    flags: numpy.ndarray[numpy.uint8]
    flexedge: numpy.ndarray[numpy.int32]
    flexedgeadr: numpy.ndarray[numpy.int32]
    flexedgenum: numpy.ndarray[numpy.int32]
    flexedgeopt: int
    flexface: numpy.ndarray[numpy.float32]
    flexfaceadr: numpy.ndarray[numpy.int32]
    flexfacenum: numpy.ndarray[numpy.int32]
    flexfaceopt: int
    flexfaceused: numpy.ndarray[numpy.int32]
    flexnormal: numpy.ndarray[numpy.float32]
    flexskinopt: int
    flextexcoord: numpy.ndarray[numpy.float32]
    flexvert: numpy.ndarray[numpy.float32]
    flexvertadr: numpy.ndarray[numpy.int32]
    flexvertnum: numpy.ndarray[numpy.int32]
    flexvertopt: int
    framergb: numpy.ndarray[numpy.float32]
    framewidth: int
    geomorder: numpy.ndarray[numpy.int32]
    maxgeom: int
    nflex: int
    ngeom: int
    nlight: int
    nskin: int
    rotate: numpy.ndarray[numpy.float32]
    scale: float
    skinfacenum: numpy.ndarray[numpy.int32]
    skinnormal: numpy.ndarray[numpy.float32]
    skinvert: numpy.ndarray[numpy.float32]
    skinvertadr: numpy.ndarray[numpy.int32]
    skinvertnum: numpy.ndarray[numpy.int32]
    stereo: int
    translate: numpy.ndarray[numpy.float32]
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._structs.MjvScene) -> None

        2. __init__(self: mujoco._structs.MjvScene, model: mujoco._structs.MjModel, maxgeom: int) -> None
        """
    @overload
    def __init__(self, model: MjModel, maxgeom: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._structs.MjvScene) -> None

        2. __init__(self: mujoco._structs.MjvScene, model: mujoco._structs.MjModel, maxgeom: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __copy__(self) -> MjvScene:
        """__copy__(self: mujoco._structs.MjvScene) -> mujoco._structs.MjvScene"""
    def __deepcopy__(self, arg0: dict) -> MjvScene:
        """__deepcopy__(self: mujoco._structs.MjvScene, arg0: dict) -> mujoco._structs.MjvScene"""
    @property
    def camera(self) -> tuple: ...
    @property
    def geoms(self) -> tuple: ...
    @property
    def lights(self) -> tuple: ...

class _MjContactList:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @overload
    def __getitem__(self, arg0: int) -> MjContact:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjContactList, arg0: int) -> mujoco._structs.MjContact

        2. __getitem__(self: mujoco._structs._MjContactList, arg0: slice) -> mujoco._structs._MjContactList
        """
    @overload
    def __getitem__(self, arg0: slice) -> _MjContactList:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjContactList, arg0: int) -> mujoco._structs.MjContact

        2. __getitem__(self: mujoco._structs._MjContactList, arg0: slice) -> mujoco._structs._MjContactList
        """
    def __iter__(self) -> typing.Iterator[MjContact]:
        """def __iter__(self) -> typing.Iterator[MjContact]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._structs._MjContactList) -> int"""
    @property
    def H(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def dim(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def dist(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def efc_address(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def elem(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def exclude(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def flex(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def frame(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def friction(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def geom(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def geom1(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def geom2(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def includemargin(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def mu(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def pos(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def solimp(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def solref(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def solreffriction(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def vert(self) -> numpy.ndarray[numpy.int32]: ...

class _MjDataActuatorViews:
    ctrl: numpy.ndarray[numpy.float64]
    force: numpy.ndarray[numpy.float64]
    length: numpy.ndarray[numpy.float64]
    moment: numpy.ndarray[numpy.float64]
    velocity: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataBodyViews:
    cacc: numpy.ndarray[numpy.float64]
    cfrc_ext: numpy.ndarray[numpy.float64]
    cfrc_int: numpy.ndarray[numpy.float64]
    cinert: numpy.ndarray[numpy.float64]
    crb: numpy.ndarray[numpy.float64]
    cvel: numpy.ndarray[numpy.float64]
    subtree_angmom: numpy.ndarray[numpy.float64]
    subtree_com: numpy.ndarray[numpy.float64]
    subtree_linvel: numpy.ndarray[numpy.float64]
    xfrc_applied: numpy.ndarray[numpy.float64]
    ximat: numpy.ndarray[numpy.float64]
    xipos: numpy.ndarray[numpy.float64]
    xmat: numpy.ndarray[numpy.float64]
    xpos: numpy.ndarray[numpy.float64]
    xquat: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataCameraViews:
    xmat: numpy.ndarray[numpy.float64]
    xpos: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataGeomViews:
    xmat: numpy.ndarray[numpy.float64]
    xpos: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataJointViews:
    cdof: numpy.ndarray[numpy.float64]
    cdof_dot: numpy.ndarray[numpy.float64]
    qLDiagInv: numpy.ndarray[numpy.float64]
    qacc: numpy.ndarray[numpy.float64]
    qacc_smooth: numpy.ndarray[numpy.float64]
    qacc_warmstart: numpy.ndarray[numpy.float64]
    qfrc_actuator: numpy.ndarray[numpy.float64]
    qfrc_applied: numpy.ndarray[numpy.float64]
    qfrc_bias: numpy.ndarray[numpy.float64]
    qfrc_constraint: numpy.ndarray[numpy.float64]
    qfrc_inverse: numpy.ndarray[numpy.float64]
    qfrc_passive: numpy.ndarray[numpy.float64]
    qfrc_smooth: numpy.ndarray[numpy.float64]
    qpos: numpy.ndarray[numpy.float64]
    qvel: numpy.ndarray[numpy.float64]
    xanchor: numpy.ndarray[numpy.float64]
    xaxis: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataLightViews:
    xdir: numpy.ndarray[numpy.float64]
    xpos: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataSensorViews:
    data: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataSiteViews:
    xmat: numpy.ndarray[numpy.float64]
    xpos: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataTendonViews:
    J: numpy.ndarray[numpy.float64]
    J_colind: numpy.ndarray[numpy.int32]
    J_rowadr: numpy.ndarray[numpy.int32]
    J_rownnz: numpy.ndarray[numpy.int32]
    length: numpy.ndarray[numpy.float64]
    velocity: numpy.ndarray[numpy.float64]
    wrapadr: numpy.ndarray[numpy.int32]
    wrapnum: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelActuatorViews:
    acc0: numpy.ndarray[numpy.float64]
    actadr: numpy.ndarray[numpy.int32]
    actlimited: numpy.ndarray[numpy.uint8]
    actnum: numpy.ndarray[numpy.int32]
    actrange: numpy.ndarray[numpy.float64]
    biasprm: numpy.ndarray[numpy.float64]
    biastype: numpy.ndarray[numpy.int32]
    cranklength: numpy.ndarray[numpy.float64]
    ctrllimited: numpy.ndarray[numpy.uint8]
    ctrlrange: numpy.ndarray[numpy.float64]
    dynprm: numpy.ndarray[numpy.float64]
    dyntype: numpy.ndarray[numpy.int32]
    forcelimited: numpy.ndarray[numpy.uint8]
    forcerange: numpy.ndarray[numpy.float64]
    gainprm: numpy.ndarray[numpy.float64]
    gaintype: numpy.ndarray[numpy.int32]
    gear: numpy.ndarray[numpy.float64]
    group: numpy.ndarray[numpy.int32]
    length0: numpy.ndarray[numpy.float64]
    lengthrange: numpy.ndarray[numpy.float64]
    trnid: numpy.ndarray[numpy.int32]
    trntype: numpy.ndarray[numpy.int32]
    user: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelBodyViews:
    dofadr: numpy.ndarray[numpy.int32]
    dofnum: numpy.ndarray[numpy.int32]
    geomadr: numpy.ndarray[numpy.int32]
    geomnum: numpy.ndarray[numpy.int32]
    inertia: numpy.ndarray[numpy.float64]
    invweight0: numpy.ndarray[numpy.float64]
    ipos: numpy.ndarray[numpy.float64]
    iquat: numpy.ndarray[numpy.float64]
    jntadr: numpy.ndarray[numpy.int32]
    jntnum: numpy.ndarray[numpy.int32]
    mass: numpy.ndarray[numpy.float64]
    mocapid: numpy.ndarray[numpy.int32]
    parentid: numpy.ndarray[numpy.int32]
    pos: numpy.ndarray[numpy.float64]
    quat: numpy.ndarray[numpy.float64]
    rootid: numpy.ndarray[numpy.int32]
    sameframe: numpy.ndarray[numpy.uint8]
    simple: numpy.ndarray[numpy.uint8]
    subtreemass: numpy.ndarray[numpy.float64]
    user: numpy.ndarray[numpy.float64]
    weldid: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelCameraViews:
    bodyid: numpy.ndarray[numpy.int32]
    fovy: numpy.ndarray[numpy.float64]
    ipd: numpy.ndarray[numpy.float64]
    mat0: numpy.ndarray[numpy.float64]
    mode: numpy.ndarray[numpy.int32]
    pos: numpy.ndarray[numpy.float64]
    pos0: numpy.ndarray[numpy.float64]
    poscom0: numpy.ndarray[numpy.float64]
    quat: numpy.ndarray[numpy.float64]
    targetbodyid: numpy.ndarray[numpy.int32]
    user: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelEqualityViews:
    active0: numpy.ndarray[numpy.uint8]
    data: numpy.ndarray[numpy.float64]
    obj1id: numpy.ndarray[numpy.int32]
    obj2id: numpy.ndarray[numpy.int32]
    solimp: numpy.ndarray[numpy.float64]
    solref: numpy.ndarray[numpy.float64]
    type: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelExcludeViews:
    signature: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelGeomViews:
    bodyid: numpy.ndarray[numpy.int32]
    conaffinity: numpy.ndarray[numpy.int32]
    condim: numpy.ndarray[numpy.int32]
    contype: numpy.ndarray[numpy.int32]
    dataid: numpy.ndarray[numpy.int32]
    friction: numpy.ndarray[numpy.float64]
    gap: numpy.ndarray[numpy.float64]
    group: numpy.ndarray[numpy.int32]
    margin: numpy.ndarray[numpy.float64]
    matid: numpy.ndarray[numpy.int32]
    pos: numpy.ndarray[numpy.float64]
    priority: numpy.ndarray[numpy.int32]
    quat: numpy.ndarray[numpy.float64]
    rbound: numpy.ndarray[numpy.float64]
    rgba: numpy.ndarray[numpy.float32]
    sameframe: numpy.ndarray[numpy.uint8]
    size: numpy.ndarray[numpy.float64]
    solimp: numpy.ndarray[numpy.float64]
    solmix: numpy.ndarray[numpy.float64]
    solref: numpy.ndarray[numpy.float64]
    type: numpy.ndarray[numpy.int32]
    user: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelHfieldViews:
    adr: numpy.ndarray[numpy.int32]
    data: numpy.ndarray[numpy.float32]
    ncol: numpy.ndarray[numpy.int32]
    nrow: numpy.ndarray[numpy.int32]
    size: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelJointViews:
    M0: numpy.ndarray[numpy.float64]
    Madr: numpy.ndarray[numpy.int32]
    armature: numpy.ndarray[numpy.float64]
    axis: numpy.ndarray[numpy.float64]
    bodyid: numpy.ndarray[numpy.int32]
    damping: numpy.ndarray[numpy.float64]
    dofadr: numpy.ndarray[numpy.int32]
    frictionloss: numpy.ndarray[numpy.float64]
    group: numpy.ndarray[numpy.int32]
    invweight0: numpy.ndarray[numpy.float64]
    jntid: numpy.ndarray[numpy.int32]
    limited: numpy.ndarray[numpy.uint8]
    margin: numpy.ndarray[numpy.float64]
    parentid: numpy.ndarray[numpy.int32]
    pos: numpy.ndarray[numpy.float64]
    qpos0: numpy.ndarray[numpy.float64]
    qpos_spring: numpy.ndarray[numpy.float64]
    qposadr: numpy.ndarray[numpy.int32]
    range: numpy.ndarray[numpy.float64]
    simplenum: numpy.ndarray[numpy.int32]
    solimp: numpy.ndarray[numpy.float64]
    solref: numpy.ndarray[numpy.float64]
    stiffness: numpy.ndarray[numpy.float64]
    type: numpy.ndarray[numpy.int32]
    user: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelKeyframeViews:
    act: numpy.ndarray[numpy.float64]
    ctrl: numpy.ndarray[numpy.float64]
    mpos: numpy.ndarray[numpy.float64]
    mquat: numpy.ndarray[numpy.float64]
    qpos: numpy.ndarray[numpy.float64]
    qvel: numpy.ndarray[numpy.float64]
    time: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelLightViews:
    active: numpy.ndarray[numpy.uint8]
    ambient: numpy.ndarray[numpy.float32]
    attenuation: numpy.ndarray[numpy.float32]
    bodyid: numpy.ndarray[numpy.int32]
    castshadow: numpy.ndarray[numpy.uint8]
    cutoff: numpy.ndarray[numpy.float32]
    diffuse: numpy.ndarray[numpy.float32]
    dir: numpy.ndarray[numpy.float64]
    dir0: numpy.ndarray[numpy.float64]
    directional: numpy.ndarray[numpy.uint8]
    exponent: numpy.ndarray[numpy.float32]
    mode: numpy.ndarray[numpy.int32]
    pos: numpy.ndarray[numpy.float64]
    pos0: numpy.ndarray[numpy.float64]
    poscom0: numpy.ndarray[numpy.float64]
    specular: numpy.ndarray[numpy.float32]
    targetbodyid: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelMaterialViews:
    emission: numpy.ndarray[numpy.float32]
    reflectance: numpy.ndarray[numpy.float32]
    rgba: numpy.ndarray[numpy.float32]
    shininess: numpy.ndarray[numpy.float32]
    specular: numpy.ndarray[numpy.float32]
    texid: numpy.ndarray[numpy.int32]
    texrepeat: numpy.ndarray[numpy.float32]
    texuniform: numpy.ndarray[numpy.uint8]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelMeshViews:
    faceadr: numpy.ndarray[numpy.int32]
    facenum: numpy.ndarray[numpy.int32]
    graphadr: numpy.ndarray[numpy.int32]
    texcoordadr: numpy.ndarray[numpy.int32]
    vertadr: numpy.ndarray[numpy.int32]
    vertnum: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelNumericViews:
    adr: numpy.ndarray[numpy.int32]
    data: numpy.ndarray[numpy.float64]
    size: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelPairViews:
    dim: numpy.ndarray[numpy.int32]
    friction: numpy.ndarray[numpy.float64]
    gap: numpy.ndarray[numpy.float64]
    geom1: numpy.ndarray[numpy.int32]
    geom2: numpy.ndarray[numpy.int32]
    margin: numpy.ndarray[numpy.float64]
    signature: numpy.ndarray[numpy.int32]
    solimp: numpy.ndarray[numpy.float64]
    solref: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelSensorViews:
    adr: numpy.ndarray[numpy.int32]
    cutoff: numpy.ndarray[numpy.float64]
    datatype: numpy.ndarray[numpy.int32]
    dim: numpy.ndarray[numpy.int32]
    needstage: numpy.ndarray[numpy.int32]
    noise: numpy.ndarray[numpy.float64]
    objid: numpy.ndarray[numpy.int32]
    objtype: numpy.ndarray[numpy.int32]
    refid: numpy.ndarray[numpy.int32]
    reftype: numpy.ndarray[numpy.int32]
    type: numpy.ndarray[numpy.int32]
    user: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelSiteViews:
    bodyid: numpy.ndarray[numpy.int32]
    group: numpy.ndarray[numpy.int32]
    matid: numpy.ndarray[numpy.int32]
    pos: numpy.ndarray[numpy.float64]
    quat: numpy.ndarray[numpy.float64]
    rgba: numpy.ndarray[numpy.float32]
    sameframe: numpy.ndarray[numpy.uint8]
    size: numpy.ndarray[numpy.float64]
    type: numpy.ndarray[numpy.int32]
    user: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelSkinViews:
    boneadr: numpy.ndarray[numpy.int32]
    bonenum: numpy.ndarray[numpy.int32]
    faceadr: numpy.ndarray[numpy.int32]
    facenum: numpy.ndarray[numpy.int32]
    inflate: numpy.ndarray[numpy.float32]
    matid: numpy.ndarray[numpy.int32]
    rgba: numpy.ndarray[numpy.float32]
    texcoordadr: numpy.ndarray[numpy.int32]
    vertadr: numpy.ndarray[numpy.int32]
    vertnum: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelTendonViews:
    _adr: numpy.ndarray[numpy.int32]
    _damping: numpy.ndarray[numpy.float64]
    _frictionloss: numpy.ndarray[numpy.float64]
    _group: numpy.ndarray[numpy.int32]
    _invweight0: numpy.ndarray[numpy.float64]
    _length0: numpy.ndarray[numpy.float64]
    _lengthspring: numpy.ndarray[numpy.float64]
    _limited: numpy.ndarray[numpy.uint8]
    _margin: numpy.ndarray[numpy.float64]
    _matid: numpy.ndarray[numpy.int32]
    _num: numpy.ndarray[numpy.int32]
    _range: numpy.ndarray[numpy.float64]
    _rgba: numpy.ndarray[numpy.float32]
    _solimp_fri: numpy.ndarray[numpy.float64]
    _solimp_lim: numpy.ndarray[numpy.float64]
    _solref_fri: numpy.ndarray[numpy.float64]
    _solref_lim: numpy.ndarray[numpy.float64]
    _stiffness: numpy.ndarray[numpy.float64]
    _user: numpy.ndarray[numpy.float64]
    _width: numpy.ndarray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelTextureViews:
    adr: numpy.ndarray[numpy.int32]
    data: numpy.ndarray[numpy.uint8]
    height: numpy.ndarray[numpy.int32]
    nchannel: numpy.ndarray[numpy.int32]
    type: numpy.ndarray[numpy.int32]
    width: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelTupleViews:
    adr: numpy.ndarray[numpy.int32]
    objid: numpy.ndarray[numpy.int32]
    objprm: numpy.ndarray[numpy.float64]
    objtype: numpy.ndarray[numpy.int32]
    size: numpy.ndarray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjSolverStatList:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @overload
    def __getitem__(self, arg0: int) -> MjSolverStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjSolverStatList, arg0: int) -> mujoco._structs.MjSolverStat

        2. __getitem__(self: mujoco._structs._MjSolverStatList, arg0: slice) -> mujoco._structs._MjSolverStatList
        """
    @overload
    def __getitem__(self, arg0: slice) -> _MjSolverStatList:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjSolverStatList, arg0: int) -> mujoco._structs.MjSolverStat

        2. __getitem__(self: mujoco._structs._MjSolverStatList, arg0: slice) -> mujoco._structs._MjSolverStatList
        """
    def __iter__(self) -> typing.Iterator[MjSolverStat]:
        """def __iter__(self) -> typing.Iterator[MjSolverStat]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._structs._MjSolverStatList) -> int"""
    @property
    def gradient(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def improvement(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def lineslope(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def nactive(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def nchange(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def neval(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def nupdate(self) -> numpy.ndarray[numpy.int32]: ...

class _MjTimerStatList:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @overload
    def __getitem__(self, arg0: int) -> MjTimerStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: int) -> mujoco._structs.MjTimerStat

        2. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: mujoco._enums.mjtTimer) -> mujoco._structs.MjTimerStat

        3. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: slice) -> mujoco._structs._MjTimerStatList
        """
    @overload
    def __getitem__(self, arg0: mujoco._enums.mjtTimer) -> MjTimerStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: int) -> mujoco._structs.MjTimerStat

        2. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: mujoco._enums.mjtTimer) -> mujoco._structs.MjTimerStat

        3. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: slice) -> mujoco._structs._MjTimerStatList
        """
    @overload
    def __getitem__(self, arg0: slice) -> _MjTimerStatList:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: int) -> mujoco._structs.MjTimerStat

        2. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: mujoco._enums.mjtTimer) -> mujoco._structs.MjTimerStat

        3. __getitem__(self: mujoco._structs._MjTimerStatList, arg0: slice) -> mujoco._structs._MjTimerStatList
        """
    def __iter__(self) -> typing.Iterator[MjTimerStat]:
        """def __iter__(self) -> typing.Iterator[MjTimerStat]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._structs._MjTimerStatList) -> int"""
    @property
    def duration(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def number(self) -> numpy.ndarray[numpy.int32]: ...

class _MjWarningStatList:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __eq__(self, arg0: object) -> bool:
        """__eq__(self: object, arg0: object) -> bool"""
    @overload
    def __getitem__(self, arg0: int) -> MjWarningStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: int) -> mujoco._structs.MjWarningStat

        2. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: mujoco._enums.mjtWarning) -> mujoco._structs.MjWarningStat

        3. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: slice) -> mujoco._structs._MjWarningStatList
        """
    @overload
    def __getitem__(self, arg0: mujoco._enums.mjtWarning) -> MjWarningStat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: int) -> mujoco._structs.MjWarningStat

        2. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: mujoco._enums.mjtWarning) -> mujoco._structs.MjWarningStat

        3. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: slice) -> mujoco._structs._MjWarningStatList
        """
    @overload
    def __getitem__(self, arg0: slice) -> _MjWarningStatList:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: int) -> mujoco._structs.MjWarningStat

        2. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: mujoco._enums.mjtWarning) -> mujoco._structs.MjWarningStat

        3. __getitem__(self: mujoco._structs._MjWarningStatList, arg0: slice) -> mujoco._structs._MjWarningStatList
        """
    def __iter__(self) -> typing.Iterator[MjWarningStat]:
        """def __iter__(self) -> typing.Iterator[MjWarningStat]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._structs._MjWarningStatList) -> int"""
    @property
    def lastinfo(self) -> numpy.ndarray[numpy.int32]: ...
    @property
    def number(self) -> numpy.ndarray[numpy.int32]: ...

def _recompile_spec_addr(arg0: int, arg1: MjModel, arg2: MjData) -> tuple:
    """_recompile_spec_addr(arg0: int, arg1: mujoco._structs.MjModel, arg2: mujoco._structs.MjData) -> tuple"""
def mjv_averageCamera(cam1: MjvGLCamera, cam2: MjvGLCamera) -> MjvGLCamera:
    """mjv_averageCamera(cam1: mujoco._structs.MjvGLCamera, cam2: mujoco._structs.MjvGLCamera) -> mujoco._structs.MjvGLCamera

    Return the average of two OpenGL cameras.
    """
