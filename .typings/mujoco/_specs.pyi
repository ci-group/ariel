import collections.abc
import flags
import mujoco._enums
import mujoco._structs
import numpy
import numpy.typing
import typing
from typing import Callable, ClassVar, overload

class MjByteVec:
    def __init__(self, arg0, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__init__(self: mujoco._specs.MjByteVec, arg0: std::byte, arg1: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def __getitem__(self, index):
        """__getitem__(self: mujoco._specs.MjByteVec, arg0: typing.SupportsInt | typing.SupportsIndex) -> std::byte"""
    def __iter__(self):
        """__iter__(self: mujoco._specs.MjByteVec) -> collections.abc.Iterator[std::byte]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjByteVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1) -> None:
        """__setitem__(self: mujoco._specs.MjByteVec, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: std::byte) -> None"""

class MjCharVec:
    def __init__(self, arg0: str, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__init__(self: mujoco._specs.MjCharVec, arg0: str, arg1: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
        """__getitem__(self: mujoco._specs.MjCharVec, arg0: typing.SupportsInt | typing.SupportsIndex) -> str"""
    def __iter__(self) -> collections.abc.Iterator[str]:
        """__iter__(self: mujoco._specs.MjCharVec) -> collections.abc.Iterator[str]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjCharVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: str) -> None:
        """__setitem__(self: mujoco._specs.MjCharVec, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: str) -> None"""

class MjDoubleVec:
    def __init__(self, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__init__(self: mujoco._specs.MjDoubleVec, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        """__getitem__(self: mujoco._specs.MjDoubleVec, arg0: typing.SupportsInt | typing.SupportsIndex) -> float"""
    def __iter__(self) -> collections.abc.Iterator[float]:
        """__iter__(self: mujoco._specs.MjDoubleVec) -> collections.abc.Iterator[float]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjDoubleVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """__setitem__(self: mujoco._specs.MjDoubleVec, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None"""

class MjFloatVec:
    def __init__(self, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__init__(self: mujoco._specs.MjFloatVec, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        """__getitem__(self: mujoco._specs.MjFloatVec, arg0: typing.SupportsInt | typing.SupportsIndex) -> float"""
    def __iter__(self) -> collections.abc.Iterator[float]:
        """__iter__(self: mujoco._specs.MjFloatVec) -> collections.abc.Iterator[float]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjFloatVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """__setitem__(self: mujoco._specs.MjFloatVec, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None"""

class MjIntVec:
    def __init__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__init__(self: mujoco._specs.MjIntVec, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> int:
        """__getitem__(self: mujoco._specs.MjIntVec, arg0: typing.SupportsInt | typing.SupportsIndex) -> int"""
    def __iter__(self) -> collections.abc.Iterator[int]:
        """__iter__(self: mujoco._specs.MjIntVec) -> collections.abc.Iterator[int]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjIntVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__setitem__(self: mujoco._specs.MjIntVec, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None"""

class MjOption:
    ccd_iterations: int
    ccd_tolerance: float
    cone: int
    density: float
    disableactuator: int
    disableflags: int
    enableflags: int
    gravity: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    impratio: float
    integrator: int
    iterations: int
    jacobian: int
    ls_iterations: int
    ls_tolerance: float
    magnetic: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    noslip_iterations: int
    noslip_tolerance: float
    o_friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    o_margin: float
    o_solimp: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    o_solref: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    sdf_initpoints: int
    sdf_iterations: int
    sleep_tolerance: float
    solver: int
    timestep: float
    tolerance: float
    viscosity: float
    wind: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjSpec:
    from_zip: ClassVar[Callable] = ...
    to_zip: ClassVar[Callable] = ...
    assets: dict
    authored: MjsAuthored
    comment: str
    compiler: MjsCompiler
    copy_during_attach: None
    hasImplicitPluginElem: bool
    memory: int
    meshdir: str
    modelfiledir: str
    modelname: str
    nconmax: int
    nemax: int
    njmax: int
    nkey: int
    nstack: int
    nuser_actuator: int
    nuser_body: int
    nuser_cam: int
    nuser_geom: int
    nuser_jnt: int
    nuser_sensor: int
    nuser_site: int
    nuser_tendon: int
    nuserdata: int
    option: MjOption
    override_assets: bool
    stat: MjStatistic
    strippath: bool
    texturedir: str
    visual: MjVisual
    def __init__(self) -> None:
        """__init__(self: mujoco._specs.MjSpec) -> None"""
    def activate_plugin(self, name: str) -> None:
        """activate_plugin(self: mujoco._specs.MjSpec, name: str) -> None"""
    def actuator(self, arg0: str) -> MjsActuator:
        """actuator(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsActuator"""
    def add_actuator(self, default: MjsDefault = ..., name: str | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., gaintype: typing.SupportsInt | typing.SupportsIndex | None = ..., gainprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., biastype: typing.SupportsInt | typing.SupportsIndex | None = ..., biasprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., dyntype: typing.SupportsInt | typing.SupportsIndex | None = ..., dynprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., actdim: typing.SupportsInt | typing.SupportsIndex | None = ..., ctrlspec: typing.SupportsInt | typing.SupportsIndex | None = ..., velrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., ffrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., actearly: typing.SupportsInt | typing.SupportsIndex | None = ..., trntype: typing.SupportsInt | typing.SupportsIndex | None = ..., gear: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., target: str | None = ..., refsite: str | None = ..., slidersite: str | None = ..., cranklength: typing.SupportsFloat | typing.SupportsIndex | None = ..., lengthrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., inheritrange: typing.SupportsFloat | typing.SupportsIndex | None = ..., damping: object | None = ..., armature: typing.SupportsFloat | typing.SupportsIndex | None = ..., ctrllimited: typing.SupportsInt | typing.SupportsIndex | None = ..., ctrlrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., forcelimited: typing.SupportsInt | typing.SupportsIndex | None = ..., forcerange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., actlimited: typing.SupportsInt | typing.SupportsIndex | None = ..., actrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., nsample: typing.SupportsInt | typing.SupportsIndex | None = ..., interp: typing.SupportsInt | typing.SupportsIndex | None = ..., delay: typing.SupportsFloat | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., plugin: MjsPlugin | None = ..., info: str | None = ...) -> MjsActuator:
        """add_actuator(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, gaintype: typing.SupportsInt | typing.SupportsIndex | None = None, gainprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, biastype: typing.SupportsInt | typing.SupportsIndex | None = None, biasprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, dyntype: typing.SupportsInt | typing.SupportsIndex | None = None, dynprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actdim: typing.SupportsInt | typing.SupportsIndex | None = None, ctrlspec: typing.SupportsInt | typing.SupportsIndex | None = None, velrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ffrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actearly: typing.SupportsInt | typing.SupportsIndex | None = None, trntype: typing.SupportsInt | typing.SupportsIndex | None = None, gear: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, target: str | None = None, refsite: str | None = None, slidersite: str | None = None, cranklength: typing.SupportsFloat | typing.SupportsIndex | None = None, lengthrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inheritrange: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: object | None = None, armature: typing.SupportsFloat | typing.SupportsIndex | None = None, ctrllimited: typing.SupportsInt | typing.SupportsIndex | None = None, ctrlrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, forcelimited: typing.SupportsInt | typing.SupportsIndex | None = None, forcerange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actlimited: typing.SupportsInt | typing.SupportsIndex | None = None, actrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, nsample: typing.SupportsInt | typing.SupportsIndex | None = None, interp: typing.SupportsInt | typing.SupportsIndex | None = None, delay: typing.SupportsFloat | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> mujoco._specs.MjsActuator


              Add actuator to spec.

              Args:
                name: str
                type: int
                gaintype: int
                gainprm: list[float]
                biastype: int
                biasprm: list[float]
                dyntype: int
                dynprm: list[float]
                actdim: int
                ctrlspec: int
                velrange: list[float]
                ffrange: list[float]
                actearly: int
                trntype: int
                gear: list[float]
                target: str
                refsite: str
                slidersite: str
                cranklength: float
                lengthrange: list[float]
                inheritrange: float
                damping: Optional[list[float]]
                armature: float
                ctrllimited: int
                ctrlrange: list[float]
                forcelimited: int
                forcerange: list[float]
                actlimited: int
                actrange: list[float]
                group: int
                nsample: int
                interp: int
                delay: float
                userdata: list[float]
                plugin: MjsPlugin
                info: str
      
        """
    def add_default(self, arg0: str, arg1: MjsDefault) -> MjsDefault:
        """add_default(self: mujoco._specs.MjSpec, arg0: str, arg1: mujoco._specs.MjsDefault) -> mujoco._specs.MjsDefault"""
    def add_equality(self, default: MjsDefault = ..., name: str | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., active: typing.SupportsInt | typing.SupportsIndex | None = ..., name1: str | None = ..., name2: str | None = ..., objtype: typing.SupportsInt | typing.SupportsIndex | None = ..., solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsEquality:
        """add_equality(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, active: typing.SupportsInt | typing.SupportsIndex | None = None, name1: str | None = None, name2: str | None = None, objtype: typing.SupportsInt | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsEquality


              Add equality to spec.

              Args:
                name: str
                type: int
                data: list[float]
                active: int
                name1: str
                name2: str
                objtype: int
                solref: list[float]
                solimp: list[float]
                info: str
      
        """
    def add_exclude(self, name: str | None = ..., bodyname1: str | None = ..., bodyname2: str | None = ..., info: str | None = ...) -> MjsExclude:
        """add_exclude(self: mujoco._specs.MjSpec, name: str | None = None, bodyname1: str | None = None, bodyname2: str | None = None, info: str | None = None) -> mujoco._specs.MjsExclude


              Add exclude to spec.

              Args:
                name: str
                bodyname1: str
                bodyname2: str
                info: str
      
        """
    def add_flex(self, name: str | None = ..., contype: typing.SupportsInt | typing.SupportsIndex | None = ..., conaffinity: typing.SupportsInt | typing.SupportsIndex | None = ..., condim: typing.SupportsInt | typing.SupportsIndex | None = ..., priority: typing.SupportsInt | typing.SupportsIndex | None = ..., friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solmix: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., margin: typing.SupportsFloat | typing.SupportsIndex | None = ..., gap: typing.SupportsFloat | typing.SupportsIndex | None = ..., dim: typing.SupportsInt | typing.SupportsIndex | None = ..., radius: typing.SupportsFloat | typing.SupportsIndex | None = ..., size: object | None = ..., flatskin: typing.SupportsInt | typing.SupportsIndex | None = ..., selfcollide: typing.SupportsInt | typing.SupportsIndex | None = ..., passive: typing.SupportsInt | typing.SupportsIndex | None = ..., activelayers: typing.SupportsInt | typing.SupportsIndex | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., edgestiffness: typing.SupportsFloat | typing.SupportsIndex | None = ..., edgedamping: typing.SupportsFloat | typing.SupportsIndex | None = ..., rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., material: collections.abc.Sequence[str] | None = ..., young: typing.SupportsFloat | typing.SupportsIndex | None = ..., poisson: typing.SupportsFloat | typing.SupportsIndex | None = ..., shear: typing.SupportsFloat | typing.SupportsIndex | None = ..., damping: typing.SupportsFloat | typing.SupportsIndex | None = ..., thickness: typing.SupportsFloat | typing.SupportsIndex | None = ..., stressyield: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., elastic2d: typing.SupportsInt | typing.SupportsIndex | None = ..., elastic3d: typing.SupportsInt | typing.SupportsIndex | None = ..., section: typing.SupportsInt | typing.SupportsIndex | None = ..., sectionsize: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., cellcount: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., order: typing.SupportsInt | typing.SupportsIndex | None = ..., nodebody: collections.abc.Sequence[str] | None = ..., vertbody: collections.abc.Sequence[str] | None = ..., node: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., vert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., elem: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., texcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., elemtexcoord: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsFlex:
        """add_flex(self: mujoco._specs.MjSpec, name: str | None = None, contype: typing.SupportsInt | typing.SupportsIndex | None = None, conaffinity: typing.SupportsInt | typing.SupportsIndex | None = None, condim: typing.SupportsInt | typing.SupportsIndex | None = None, priority: typing.SupportsInt | typing.SupportsIndex | None = None, friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solmix: typing.SupportsFloat | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, gap: typing.SupportsFloat | typing.SupportsIndex | None = None, dim: typing.SupportsInt | typing.SupportsIndex | None = None, radius: typing.SupportsFloat | typing.SupportsIndex | None = None, size: object | None = None, flatskin: typing.SupportsInt | typing.SupportsIndex | None = None, selfcollide: typing.SupportsInt | typing.SupportsIndex | None = None, passive: typing.SupportsInt | typing.SupportsIndex | None = None, activelayers: typing.SupportsInt | typing.SupportsIndex | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, edgestiffness: typing.SupportsFloat | typing.SupportsIndex | None = None, edgedamping: typing.SupportsFloat | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, material: collections.abc.Sequence[str] | None = None, young: typing.SupportsFloat | typing.SupportsIndex | None = None, poisson: typing.SupportsFloat | typing.SupportsIndex | None = None, shear: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: typing.SupportsFloat | typing.SupportsIndex | None = None, thickness: typing.SupportsFloat | typing.SupportsIndex | None = None, stressyield: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, elastic2d: typing.SupportsInt | typing.SupportsIndex | None = None, elastic3d: typing.SupportsInt | typing.SupportsIndex | None = None, section: typing.SupportsInt | typing.SupportsIndex | None = None, sectionsize: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, cellcount: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, order: typing.SupportsInt | typing.SupportsIndex | None = None, nodebody: collections.abc.Sequence[str] | None = None, vertbody: collections.abc.Sequence[str] | None = None, node: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, vert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, elem: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, texcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, elemtexcoord: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsFlex


              Add flex to spec.

              Args:
                name: str
                contype: int
                conaffinity: int
                condim: int
                priority: int
                friction: list[float]
                solmix: float
                solref: list[float]
                solimp: list[float]
                margin: float
                gap: float
                dim: int
                radius: float
                size: Optional[list[float]]
                flatskin: int
                selfcollide: int
                passive: int
                activelayers: int
                group: int
                edgestiffness: float
                edgedamping: float
                rgba: list[float]
                material: list[str]
                young: float
                poisson: float
                shear: float
                damping: float
                thickness: float
                stressyield: list[float]
                elastic2d: int
                elastic3d: int
                section: int
                sectionsize: list[float]
                cellcount: list[float]
                order: int
                nodebody: list[str]
                vertbody: list[str]
                node: list[float]
                vert: list[float]
                elem: list[int]
                texcoord: list[float]
                elemtexcoord: list[int]
                info: str
      
        """
    def add_hfield(self, name: str | None = ..., content_type: str | None = ..., file: str | None = ..., size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., nrow: typing.SupportsInt | typing.SupportsIndex | None = ..., ncol: typing.SupportsInt | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsHField:
        """add_hfield(self: mujoco._specs.MjSpec, name: str | None = None, content_type: str | None = None, file: str | None = None, size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, nrow: typing.SupportsInt | typing.SupportsIndex | None = None, ncol: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsHField


              Add hfield to spec.

              Args:
                name: str
                content_type: str
                file: str
                size: list[float]
                nrow: int
                ncol: int
                userdata: list[float]
                info: str
      
        """
    def add_key(self, name: str | None = ..., time: typing.SupportsFloat | typing.SupportsIndex | None = ..., qpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., qvel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., act: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., ctrl: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., plastic: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsKey:
        """add_key(self: mujoco._specs.MjSpec, name: str | None = None, time: typing.SupportsFloat | typing.SupportsIndex | None = None, qpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, qvel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, act: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ctrl: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plastic: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsKey


              Add key to spec.

              Args:
                name: str
                time: float
                qpos: list[float]
                qvel: list[float]
                act: list[float]
                mpos: list[float]
                mquat: list[float]
                ctrl: list[float]
                plastic: list[float]
                info: str
      
        """
    def add_material(self, default: MjsDefault = ..., name: str | None = ..., textures: collections.abc.Sequence[str] | None = ..., texuniform: typing.SupportsInt | typing.SupportsIndex | None = ..., texrepeat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., emission: typing.SupportsFloat | typing.SupportsIndex | None = ..., specular: typing.SupportsFloat | typing.SupportsIndex | None = ..., shininess: typing.SupportsFloat | typing.SupportsIndex | None = ..., reflectance: typing.SupportsFloat | typing.SupportsIndex | None = ..., metallic: typing.SupportsFloat | typing.SupportsIndex | None = ..., roughness: typing.SupportsFloat | typing.SupportsIndex | None = ..., rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsMaterial:
        """add_material(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, name: str | None = None, textures: collections.abc.Sequence[str] | None = None, texuniform: typing.SupportsInt | typing.SupportsIndex | None = None, texrepeat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, emission: typing.SupportsFloat | typing.SupportsIndex | None = None, specular: typing.SupportsFloat | typing.SupportsIndex | None = None, shininess: typing.SupportsFloat | typing.SupportsIndex | None = None, reflectance: typing.SupportsFloat | typing.SupportsIndex | None = None, metallic: typing.SupportsFloat | typing.SupportsIndex | None = None, roughness: typing.SupportsFloat | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsMaterial


              Add material to spec.

              Args:
                name: str
                textures: list[str]
                texuniform: int
                texrepeat: list[float]
                emission: float
                specular: float
                shininess: float
                reflectance: float
                metallic: float
                roughness: float
                rgba: list[float]
                info: str
      
        """
    def add_mesh(self, default: MjsDefault = ..., name: str | None = ..., content_type: str | None = ..., file: str | None = ..., refpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., refquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., scale: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., inertia: typing.SupportsInt | typing.SupportsIndex | None = ..., smoothnormal: typing.SupportsInt | typing.SupportsIndex | None = ..., needsdf: typing.SupportsInt | typing.SupportsIndex | None = ..., maxhullvert: typing.SupportsInt | typing.SupportsIndex | None = ..., uservert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., usernormal: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., usertexcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., userface: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., userfacenormal: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., userfacetexcoord: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., usernode: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., usertet: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., plugin: MjsPlugin | None = ..., material: str | None = ..., octree_maxdepth: typing.SupportsInt | typing.SupportsIndex | None = ..., info: str | None = ...) -> MjsMesh:
        """add_mesh(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, name: str | None = None, content_type: str | None = None, file: str | None = None, refpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, refquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, scale: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inertia: typing.SupportsInt | typing.SupportsIndex | None = None, smoothnormal: typing.SupportsInt | typing.SupportsIndex | None = None, needsdf: typing.SupportsInt | typing.SupportsIndex | None = None, maxhullvert: typing.SupportsInt | typing.SupportsIndex | None = None, uservert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, usernormal: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, usertexcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userface: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, userfacenormal: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, userfacetexcoord: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, usernode: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, usertet: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, material: str | None = None, octree_maxdepth: typing.SupportsInt | typing.SupportsIndex | None = None, info: str | None = None) -> mujoco._specs.MjsMesh


              Add mesh to spec.

              Args:
                name: str
                content_type: str
                file: str
                refpos: list[float]
                refquat: list[float]
                scale: list[float]
                inertia: int
                smoothnormal: int
                needsdf: int
                maxhullvert: int
                uservert: list[float]
                usernormal: list[float]
                usertexcoord: list[float]
                userface: list[int]
                userfacenormal: list[int]
                userfacetexcoord: list[int]
                usernode: list[float]
                usertet: list[int]
                plugin: MjsPlugin
                material: str
                octree_maxdepth: int
                info: str
      
        """
    def add_numeric(self, name: str | None = ..., data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., size: typing.SupportsInt | typing.SupportsIndex | None = ..., info: str | None = ...) -> MjsNumeric:
        """add_numeric(self: mujoco._specs.MjSpec, name: str | None = None, data: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: typing.SupportsInt | typing.SupportsIndex | None = None, info: str | None = None) -> mujoco._specs.MjsNumeric


              Add numeric to spec.

              Args:
                name: str
                data: list[float]
                size: int
                info: str
      
        """
    def add_pair(self, default: MjsDefault = ..., name: str | None = ..., geomname1: str | None = ..., geomname2: str | None = ..., condim: typing.SupportsInt | typing.SupportsIndex | None = ..., solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solreffriction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., margin: typing.SupportsFloat | typing.SupportsIndex | None = ..., gap: typing.SupportsFloat | typing.SupportsIndex | None = ..., adhesion: typing.SupportsFloat | typing.SupportsIndex | None = ..., friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsPair:
        """add_pair(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, name: str | None = None, geomname1: str | None = None, geomname2: str | None = None, condim: typing.SupportsInt | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solreffriction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, gap: typing.SupportsFloat | typing.SupportsIndex | None = None, adhesion: typing.SupportsFloat | typing.SupportsIndex | None = None, friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsPair


              Add pair to spec.

              Args:
                name: str
                geomname1: str
                geomname2: str
                condim: int
                solref: list[float]
                solreffriction: list[float]
                solimp: list[float]
                margin: float
                gap: float
                adhesion: float
                friction: list[float]
                info: str
      
        """
    def add_plugin(self, name: str | None = ..., plugin_name: str | None = ..., active: typing.SupportsInt | typing.SupportsIndex | None = ..., info: str | None = ...) -> MjsPlugin:
        """add_plugin(self: mujoco._specs.MjSpec, name: str | None = None, plugin_name: str | None = None, active: typing.SupportsInt | typing.SupportsIndex | None = None, info: str | None = None) -> mujoco._specs.MjsPlugin


              Add plugin to spec.

              Args:
                name: str
                plugin_name: str
                active: int
                info: str
      
        """
    def add_sensor(self, name: str | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., objtype: typing.SupportsInt | typing.SupportsIndex | None = ..., objname: str | None = ..., reftype: typing.SupportsInt | typing.SupportsIndex | None = ..., refname: str | None = ..., intprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., datatype: typing.SupportsInt | typing.SupportsIndex | None = ..., needstage: typing.SupportsInt | typing.SupportsIndex | None = ..., dim: typing.SupportsInt | typing.SupportsIndex | None = ..., cutoff: typing.SupportsFloat | typing.SupportsIndex | None = ..., noise: typing.SupportsFloat | typing.SupportsIndex | None = ..., nsample: typing.SupportsInt | typing.SupportsIndex | None = ..., interp: typing.SupportsInt | typing.SupportsIndex | None = ..., delay: typing.SupportsFloat | typing.SupportsIndex | None = ..., interval: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., plugin: MjsPlugin | None = ..., info: str | None = ...) -> MjsSensor:
        """add_sensor(self: mujoco._specs.MjSpec, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, objtype: typing.SupportsInt | typing.SupportsIndex | None = None, objname: str | None = None, reftype: typing.SupportsInt | typing.SupportsIndex | None = None, refname: str | None = None, intprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, datatype: typing.SupportsInt | typing.SupportsIndex | None = None, needstage: typing.SupportsInt | typing.SupportsIndex | None = None, dim: typing.SupportsInt | typing.SupportsIndex | None = None, cutoff: typing.SupportsFloat | typing.SupportsIndex | None = None, noise: typing.SupportsFloat | typing.SupportsIndex | None = None, nsample: typing.SupportsInt | typing.SupportsIndex | None = None, interp: typing.SupportsInt | typing.SupportsIndex | None = None, delay: typing.SupportsFloat | typing.SupportsIndex | None = None, interval: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> mujoco._specs.MjsSensor


              Add sensor to spec.

              Args:
                name: str
                type: int
                objtype: int
                objname: str
                reftype: int
                refname: str
                intprm: list[float]
                datatype: int
                needstage: int
                dim: int
                cutoff: float
                noise: float
                nsample: int
                interp: int
                delay: float
                interval: list[float]
                userdata: list[float]
                plugin: MjsPlugin
                info: str
      
        """
    def add_skin(self, name: str | None = ..., file: str | None = ..., material: str | None = ..., rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., inflate: typing.SupportsFloat | typing.SupportsIndex | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., vert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., texcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., face: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., bodyname: collections.abc.Sequence[str] | None = ..., bindpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., bindquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., vertid: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]] | None = ..., vertweight: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]] | None = ..., info: str | None = ...) -> MjsSkin:
        """add_skin(self: mujoco._specs.MjSpec, name: str | None = None, file: str | None = None, material: str | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inflate: typing.SupportsFloat | typing.SupportsIndex | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, vert: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, texcoord: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, face: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, bodyname: collections.abc.Sequence[str] | None = None, bindpos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, bindquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, vertid: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]] | None = None, vertweight: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]] | None = None, info: str | None = None) -> mujoco._specs.MjsSkin


              Add skin to spec.

              Args:
                name: str
                file: str
                material: str
                rgba: list[float]
                inflate: float
                group: int
                vert: list[float]
                texcoord: list[float]
                face: list[int]
                bodyname: list[str]
                bindpos: list[float]
                bindquat: list[float]
                vertid: list[list[int]]
                vertweight: list[list[float]]
                info: str
      
        """
    def add_tendon(self, default: MjsDefault = ..., name: str | None = ..., stiffness: object | None = ..., springlength: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., springyield: typing.SupportsFloat | typing.SupportsIndex | None = ..., damping: object | None = ..., frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., armature: typing.SupportsFloat | typing.SupportsIndex | None = ..., limited: typing.SupportsInt | typing.SupportsIndex | None = ..., actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = ..., range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., margin: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., material: str | None = ..., width: typing.SupportsFloat | typing.SupportsIndex | None = ..., rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsTendon:
        """add_tendon(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, name: str | None = None, stiffness: object | None = None, springlength: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, springyield: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: object | None = None, frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, armature: typing.SupportsFloat | typing.SupportsIndex | None = None, limited: typing.SupportsInt | typing.SupportsIndex | None = None, actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = None, range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, material: str | None = None, width: typing.SupportsFloat | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsTendon


              Add tendon to spec.

              Args:
                name: str
                stiffness: Optional[list[float]]
                springlength: list[float]
                springyield: float
                damping: Optional[list[float]]
                frictionloss: float
                solref_friction: list[float]
                solimp_friction: list[float]
                armature: float
                limited: int
                actfrclimited: int
                range: list[float]
                actfrcrange: list[float]
                margin: float
                solref_limit: list[float]
                solimp_limit: list[float]
                material: str
                width: float
                rgba: list[float]
                group: int
                userdata: list[float]
                info: str
      
        """
    def add_text(self, name: str | None = ..., data: str | None = ..., info: str | None = ...) -> MjsText:
        """add_text(self: mujoco._specs.MjSpec, name: str | None = None, data: str | None = None, info: str | None = None) -> mujoco._specs.MjsText


              Add text to spec.

              Args:
                name: str
                data: str
                info: str
      
        """
    def add_texture(self, default: MjsDefault = ..., name: str | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., colorspace: typing.SupportsInt | typing.SupportsIndex | None = ..., builtin: typing.SupportsInt | typing.SupportsIndex | None = ..., mark: typing.SupportsInt | typing.SupportsIndex | None = ..., rgb1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., rgb2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., markrgb: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., random: typing.SupportsFloat | typing.SupportsIndex | None = ..., height: typing.SupportsInt | typing.SupportsIndex | None = ..., width: typing.SupportsInt | typing.SupportsIndex | None = ..., nchannel: typing.SupportsInt | typing.SupportsIndex | None = ..., content_type: str | None = ..., file: str | None = ..., gridsize: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., gridlayout: object = ..., cubefiles: collections.abc.Sequence[str] | None = ..., data: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., hflip: typing.SupportsInt | typing.SupportsIndex | None = ..., vflip: typing.SupportsInt | typing.SupportsIndex | None = ..., embed: typing.SupportsInt | typing.SupportsIndex | None = ..., info: str | None = ...) -> MjsTexture:
        """add_texture(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, colorspace: typing.SupportsInt | typing.SupportsIndex | None = None, builtin: typing.SupportsInt | typing.SupportsIndex | None = None, mark: typing.SupportsInt | typing.SupportsIndex | None = None, rgb1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, rgb2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, markrgb: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, random: typing.SupportsFloat | typing.SupportsIndex | None = None, height: typing.SupportsInt | typing.SupportsIndex | None = None, width: typing.SupportsInt | typing.SupportsIndex | None = None, nchannel: typing.SupportsInt | typing.SupportsIndex | None = None, content_type: str | None = None, file: str | None = None, gridsize: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, gridlayout: object = None, cubefiles: collections.abc.Sequence[str] | None = None, data: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, hflip: typing.SupportsInt | typing.SupportsIndex | None = None, vflip: typing.SupportsInt | typing.SupportsIndex | None = None, embed: typing.SupportsInt | typing.SupportsIndex | None = None, info: str | None = None) -> mujoco._specs.MjsTexture


              Add texture to spec.

              Args:
                name: str
                type: int
                colorspace: int
                builtin: int
                mark: int
                rgb1: list[float]
                rgb2: list[float]
                markrgb: list[float]
                random: float
                height: int
                width: int
                nchannel: int
                content_type: str
                file: str
                gridsize: list[float]
                gridlayout: str | list[str]
                cubefiles: list[str]
                data: list[int]
                hflip: int
                vflip: int
                embed: int
                info: str
      
        """
    def add_tuple(self, name: str | None = ..., objtype: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., objname: collections.abc.Sequence[str] | None = ..., objprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsTuple:
        """add_tuple(self: mujoco._specs.MjSpec, name: str | None = None, objtype: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, objname: collections.abc.Sequence[str] | None = None, objprm: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsTuple


              Add tuple to spec.

              Args:
                name: str
                objtype: list[int]
                objname: list[str]
                objprm: list[float]
                info: str
      
        """
    def adopt_inertial(self, body: MjsBody, vfs: MjVfs | None = ...) -> None:
        """adopt_inertial(self: mujoco._specs.MjSpec, body: mujoco._specs.MjsBody, vfs: mujoco._specs.MjVfs | None = None) -> None


            Makes the inertial which compilation infers for a body part of the spec.

            The mass, inertia and inertial frame calculated from the geoms of the body
            become its explicit inertial, and no longer follow changes to the geoms.
            Assets are read as in compile().

            Parameters
            ----------
            body : MjsBody
                A body of this spec.
            vfs : MjVfs, optional
                A VFS to read assets from. Cannot be used with assets.
    
        """
    def attach(self, child: MjSpec, prefix: str | None = ..., suffix: str | None = ..., site: object | None = ..., frame: object | None = ...) -> MjsFrame:
        """attach(self: mujoco._specs.MjSpec, child: mujoco._specs.MjSpec, prefix: str | None = None, suffix: str | None = None, site: object | None = None, frame: object | None = None) -> mujoco._specs.MjsFrame"""
    def body(self, arg0: str) -> MjsBody:
        """body(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsBody"""
    def camera(self, arg0: str) -> MjsCamera:
        """camera(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsCamera"""
    def compile(self, vfs: MjVfs | None = ...) -> object:
        """compile(self: mujoco._specs.MjSpec, vfs: mujoco._specs.MjVfs | None = None) -> object"""
    def copy(self) -> MjSpec:
        """copy(self: mujoco._specs.MjSpec) -> mujoco._specs.MjSpec"""
    def copy_back(self, model: object) -> None:
        """copy_back(self: mujoco._specs.MjSpec, model: object) -> None


            Writes to the spec what was changed in a model compiled from it.

            Each value of the model which is not the one that the spec was compiled to
            is written to the spec as what compiles to it, so that it is in the models
            which are compiled from the spec afterwards and in the XML which is saved.
            Raises ValueError, and writes nothing, if a change cannot be expressed in
            the spec.

            Parameters
            ----------
            model : MjModel
                The model which was last compiled from this spec.
    
        """
    def dedup_assets(self) -> int:
        """dedup_assets(self: mujoco._specs.MjSpec) -> int


            Merges the duplicate textures and meshes of the spec.

            Textures and meshes which would compile to the same asset, with the same
            resolved file or buffers and the same parameters, are merged: the first of
            each group keeps its name, every reference to the others is rewritten to it
            and the others are deleted. Objects referring to the deleted elements must
            not be used afterwards. Nothing is read from disk. Returns the number of
            assets merged.
    
        """
    @overload
    def delete(self, arg0: MjsDefault) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsBody) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsFrame) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsGeom) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsJoint) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsSite) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsCamera) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsLight) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsMaterial) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsMesh) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsPair) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsEquality) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsActuator) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsTendon) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsSensor) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsFlex) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsHField) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsSkin) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsTexture) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsKey) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsText) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsNumeric) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsExclude) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsTuple) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    @overload
    def delete(self, arg0: MjsPlugin) -> None:
        """delete(*args, **kwargs)
        Overloaded function.

        1. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsDefault) -> None

        2. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsBody) -> None

        3. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFrame) -> None

        4. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsGeom) -> None

        5. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsJoint) -> None

        6. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSite) -> None

        7. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsCamera) -> None

        8. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsLight) -> None

        9. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMaterial) -> None

        10. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsMesh) -> None

        11. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPair) -> None

        12. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsEquality) -> None

        13. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsActuator) -> None

        14. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTendon) -> None

        15. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSensor) -> None

        16. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsFlex) -> None

        17. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsHField) -> None

        18. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsSkin) -> None

        19. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTexture) -> None

        20. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsKey) -> None

        21. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsText) -> None

        22. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsNumeric) -> None

        23. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsExclude) -> None

        24. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsTuple) -> None

        25. delete(self: mujoco._specs.MjSpec, arg0: mujoco._specs.MjsPlugin) -> None
        """
    def discard_visual(self, vfs: MjVfs | None = ...) -> int:
        """discard_visual(self: mujoco._specs.MjSpec, vfs: mujoco._specs.MjVfs | None = None) -> int


            Discards the visual elements of the spec.

            All materials and textures are discarded, as are the geoms which do not
            collide and are not referenced, and the meshes which are then not used.
            Objects referring to the discarded elements must not be used afterwards.
            Assets are read as in compile(), and only if body inertia is inferred from
            a discarded geom. Returns the number of elements discarded.

            Parameters
            ----------
            vfs : MjVfs, optional
                A VFS to read assets from. Cannot be used with assets.
    
        """
    def encode(self, filename: str, model: object | None = ..., content_type: str | None = ...) -> int:
        """encode(self: mujoco._specs.MjSpec, filename: str, model: object | None = None, content_type: str | None = None) -> int"""
    def equality(self, arg0: str) -> MjsEquality:
        """equality(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsEquality"""
    def exclude(self, arg0: str) -> MjsExclude:
        """exclude(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsExclude"""
    def find_default(self, arg0: str) -> MjsDefault:
        """find_default(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsDefault"""
    def flex(self, arg0: str) -> MjsFlex:
        """flex(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsFlex"""
    def frame(self, arg0: str) -> MjsFrame:
        """frame(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsFrame"""
    @staticmethod
    def from_file(filename: str, include: collections.abc.Mapping[str, bytes] | None = ..., assets: dict | None = ..., vfs: MjVfs = ...) -> MjSpec:
        """from_file(filename: str, include: collections.abc.Mapping[str, bytes] | None = None, assets: dict | None = None, vfs: mujoco._specs.MjVfs = None) -> mujoco._specs.MjSpec


            Creates a spec from an XML file.

            Parameters
            ----------
            filename : str
                Path to the XML file.
            include : dict, optional
                A dictionary of xml files included by the model. The keys are file names
                and the values are file contents.
            assets : dict, optional
                A dictionary of assets to be used by the spec. The keys are asset names
                and the values are asset contents.
            vfs : MjVfs, optional
                A VFS to use for resolving includes and assets. Cannot be used with
                include or assets.
  
        """
    @staticmethod
    def from_string(xml: str, include: collections.abc.Mapping[str, bytes] | None = ..., assets: dict | None = ..., vfs: MjVfs = ...) -> MjSpec:
        """from_string(xml: str, include: collections.abc.Mapping[str, bytes] | None = None, assets: dict | None = None, vfs: mujoco._specs.MjVfs = None) -> mujoco._specs.MjSpec


            Creates a spec from an XML string.

            Parameters
            ----------
            xml : str
                XML string.
            include : dict, optional
                A dictionary of xml files included by the model. The keys are file names
                and the values are file contents.
            assets : dict, optional
                A dictionary of assets to be used by the spec. The keys are asset names
                and the values are asset contents.
            vfs : MjVfs, optional
                A VFS to use for resolving includes and assets. Cannot be used with
                include or assets.
  
        """
    def fuse_static(self, vfs: MjVfs | None = ...) -> int:
        """fuse_static(self: mujoco._specs.MjSpec, vfs: mujoco._specs.MjVfs | None = None) -> int


            Fuses the static bodies of the spec with their parents.

            Each fused body becomes a frame in its parent, holding what the body held.
            Objects referring to the fused bodies must not be used afterwards. Assets
            are read as in compile(). Returns the number of bodies fused.

            Parameters
            ----------
            vfs : MjVfs, optional
                A VFS to read assets from. Cannot be used with assets.
    
        """
    def geom(self, arg0: str) -> MjsGeom:
        """geom(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsGeom"""
    def hfield(self, arg0: str) -> MjsHField:
        """hfield(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsHField"""
    def joint(self, arg0: str) -> MjsJoint:
        """joint(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsJoint"""
    def key(self, arg0: str) -> MjsKey:
        """key(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsKey"""
    def light(self, arg0: str) -> MjsLight:
        """light(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsLight"""
    def material(self, arg0: str) -> MjsMaterial:
        """material(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsMaterial"""
    def mesh(self, arg0: str) -> MjsMesh:
        """mesh(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsMesh"""
    def numeric(self, arg0: str) -> MjsNumeric:
        """numeric(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsNumeric"""
    def pair(self, arg0: str) -> MjsPair:
        """pair(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsPair"""
    def plugin(self, arg0: str) -> MjsPlugin:
        """plugin(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsPlugin"""
    def recompile(self, m: object, d: object, vfs: MjVfs | None = ...) -> object:
        """recompile(self: mujoco._specs.MjSpec, m: object, d: object, vfs: mujoco._specs.MjVfs | None = None) -> object"""
    @staticmethod
    def resolve_orientation(*args, **kwargs):
        '''resolve_orientation(degree: bool, sequence: mujoco._specs.MjCharVec = None, orientation: mujoco._specs.MjsOrientation) -> typing.Annotated[list[float], "FixedSize(4)"]'''
    def sensor(self, arg0: str) -> MjsSensor:
        """sensor(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsSensor"""
    def site(self, arg0: str) -> MjsSite:
        """site(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsSite"""
    def skin(self, arg0: str) -> MjsSkin:
        """skin(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsSkin"""
    def tendon(self, arg0: str) -> MjsTendon:
        """tendon(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsTendon"""
    def text(self, arg0: str) -> MjsText:
        """text(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsText"""
    def texture(self, arg0: str) -> MjsTexture:
        """texture(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsTexture"""
    def to_file(self, arg0: str) -> None:
        """to_file(self: mujoco._specs.MjSpec, arg0: str) -> None"""
    def to_xml(self) -> str:
        """to_xml(self: mujoco._specs.MjSpec) -> str"""
    def tuple(self, arg0: str) -> MjsTuple:
        """tuple(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsTuple"""
    @property
    def actuators(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def bodies(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def cameras(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def default(self) -> MjsDefault:
        """(arg0: mujoco._specs.MjSpec) -> mujoco._specs.MjsDefault"""
    @property
    def equalities(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def excludes(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def flexes(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def frames(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def geoms(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def hfields(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def joints(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def keys(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def lights(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def materials(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def meshes(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def numerics(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def pairs(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def parent(self) -> MjSpec:
        """(arg0: mujoco._specs.MjSpec) -> mujoco._specs.MjSpec"""
    @property
    def plugins(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def sensors(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def sites(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def skins(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def tendons(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def texts(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def textures(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def timer(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], '[9, 1]', 'flags.writeable']:
        '''(arg0: mujoco._specs.MjSpec) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]", "flags.writeable"]'''
    @property
    def tuples(self) -> list:
        """(arg0: mujoco._specs.MjSpec) -> list"""
    @property
    def worldbody(self) -> MjsBody:
        """(arg0: mujoco._specs.MjSpec) -> mujoco._specs.MjsBody"""

class MjStatistic:
    center: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    extent: float
    meaninertia: float
    meanmass: float
    meansize: float
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjStringVec:
    def __init__(self, arg0: str, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        """__init__(self: mujoco._specs.MjStringVec, arg0: str, arg1: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
        """__getitem__(self: mujoco._specs.MjStringVec, arg0: typing.SupportsInt | typing.SupportsIndex) -> str"""
    def __iter__(self) -> collections.abc.Iterator[str]:
        """__iter__(self: mujoco._specs.MjStringVec) -> collections.abc.Iterator[str]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjStringVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: str) -> None:
        """__setitem__(self: mujoco._specs.MjStringVec, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: str) -> None"""

class MjVfs:
    def __init__(self) -> None:
        """__init__(self: mujoco._specs.MjVfs) -> None"""
    def close(self) -> None:
        """close(self: mujoco._specs.MjVfs) -> None"""
    def __contains__(self, arg0: str) -> bool:
        """__contains__(self: mujoco._specs.MjVfs, arg0: str) -> bool"""
    def __delitem__(self, arg0: str) -> None:
        """__delitem__(self: mujoco._specs.MjVfs, arg0: str) -> None"""
    def __enter__(self) -> MjVfs:
        """__enter__(self: mujoco._specs.MjVfs) -> mujoco._specs.MjVfs"""
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None:
        """__exit__(self: mujoco._specs.MjVfs, arg0: object, arg1: object, arg2: object) -> None"""
    def __setitem__(self, arg0: str, arg1: bytes) -> None:
        """__setitem__(self: mujoco._specs.MjVfs, arg0: str, arg1: bytes) -> None"""

class MjVisual:
    global_: mujoco._structs.MjVisual.Global
    headlight: MjVisualHeadlight
    map: mujoco._structs.MjVisual.Map
    quality: mujoco._structs.MjVisual.Quality
    rgba: MjVisualRgba
    scale: mujoco._structs.MjVisual.Scale
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjVisualHeadlight:
    active: int
    ambient: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    diffuse: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    specular: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjVisualRgba:
    actuator: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    actuatornegative: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    actuatorpositive: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    bv: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    bvactive: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    camera: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    com: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    connect: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    constraint: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    contactforce: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    contactfriction: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    contactgap: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    contactpoint: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    contacttorque: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    crankbroken: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    fog: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    force: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    frustum: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    haze: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    inertia: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    joint: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    light: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    rangefinder: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    selectpoint: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    slidercrank: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjsActuator:
    actdim: int
    actearly: bool
    actlimited: mujoco._enums.mjtLimited
    actrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    armature: float
    biasprm: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[10, 1]', 'flags.writeable']
    biastype: mujoco._enums.mjtBias
    classname: MjsDefault
    cranklength: float
    ctrllimited: mujoco._enums.mjtLimited
    ctrlrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    ctrlspec: int
    damping: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    delay: float
    dynprm: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[10, 1]', 'flags.writeable']
    dyntype: mujoco._enums.mjtDyn
    ffrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    forcelimited: mujoco._enums.mjtLimited
    forcerange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    gainprm: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[10, 1]', 'flags.writeable']
    gaintype: mujoco._enums.mjtGain
    gear: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]', 'flags.writeable']
    group: int
    info: str
    inheritrange: float
    interp: int
    lengthrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    name: str
    nsample: int
    plugin: MjsPlugin
    refsite: str
    slidersite: str
    target: str
    trntype: mujoco._enums.mjtTrn
    type: mujoco._enums.mjtActuator
    userdata: MjDoubleVec
    velrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def set_to_adhesion(self, gain: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """set_to_adhesion(self: mujoco._specs.MjsActuator, gain: typing.SupportsFloat | typing.SupportsIndex) -> None"""
    def set_to_cylinder(self, timeconst: typing.SupportsFloat | typing.SupportsIndex, bias: typing.SupportsFloat | typing.SupportsIndex, area: typing.SupportsFloat | typing.SupportsIndex, diameter: typing.SupportsFloat | typing.SupportsIndex = ...) -> None:
        """set_to_cylinder(self: mujoco._specs.MjsActuator, timeconst: typing.SupportsFloat | typing.SupportsIndex, bias: typing.SupportsFloat | typing.SupportsIndex, area: typing.SupportsFloat | typing.SupportsIndex, diameter: typing.SupportsFloat | typing.SupportsIndex = -1) -> None"""
    def set_to_damper(self, kv: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """set_to_damper(self: mujoco._specs.MjsActuator, kv: typing.SupportsFloat | typing.SupportsIndex) -> None"""
    def set_to_dcmotor(self, motorconst, resistance: typing.SupportsFloat | typing.SupportsIndex, nominal=..., saturation=..., inductance=..., cogging=..., controller=..., thermal=..., lugre=..., ctrlspec: typing.SupportsInt | typing.SupportsIndex = ...) -> None:
        '''set_to_dcmotor(self: mujoco._specs.MjsActuator, motorconst: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"], resistance: typing.SupportsFloat | typing.SupportsIndex, nominal: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 0.0], saturation: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 0.0], inductance: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"] = [0.0, 0.0], cogging: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 0.0], controller: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(6)"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], thermal: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(6)"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lugre: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(5)"] = [0.0, 0.0, 0.0, 0.0, 0.0], ctrlspec: typing.SupportsInt | typing.SupportsIndex = 0) -> None'''
    def set_to_intvelocity(self, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = ..., dampratio: typing.SupportsFloat | typing.SupportsIndex = ..., timeconst: typing.SupportsFloat | typing.SupportsIndex = ..., inheritrange: bool = ...) -> None:
        """set_to_intvelocity(self: mujoco._specs.MjsActuator, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = -1, dampratio: typing.SupportsFloat | typing.SupportsIndex = -1, timeconst: typing.SupportsFloat | typing.SupportsIndex = -1, inheritrange: bool = False) -> None"""
    def set_to_motor(self) -> None:
        """set_to_motor(self: mujoco._specs.MjsActuator) -> None"""
    def set_to_muscle(self, timeconst=..., tausmooth: typing.SupportsFloat | typing.SupportsIndex = ..., range=..., force: typing.SupportsFloat | typing.SupportsIndex = ..., scale: typing.SupportsFloat | typing.SupportsIndex = ..., lmin: typing.SupportsFloat | typing.SupportsIndex = ..., lmax: typing.SupportsFloat | typing.SupportsIndex = ..., vmax: typing.SupportsFloat | typing.SupportsIndex = ..., fpmax: typing.SupportsFloat | typing.SupportsIndex = ..., fvmax: typing.SupportsFloat | typing.SupportsIndex = ...) -> None:
        '''set_to_muscle(self: mujoco._specs.MjsActuator, timeconst: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"] = [-1.0, -1.0], tausmooth: typing.SupportsFloat | typing.SupportsIndex, range: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"] = [-1.0, -1.0], force: typing.SupportsFloat | typing.SupportsIndex = -1, scale: typing.SupportsFloat | typing.SupportsIndex = -1, lmin: typing.SupportsFloat | typing.SupportsIndex = -1, lmax: typing.SupportsFloat | typing.SupportsIndex = -1, vmax: typing.SupportsFloat | typing.SupportsIndex = -1, fpmax: typing.SupportsFloat | typing.SupportsIndex = -1, fvmax: typing.SupportsFloat | typing.SupportsIndex = -1) -> None'''
    def set_to_orientation(self, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = ..., dampratio: typing.SupportsFloat | typing.SupportsIndex = ..., ctrlspec: typing.SupportsInt | typing.SupportsIndex = ...) -> None:
        """set_to_orientation(self: mujoco._specs.MjsActuator, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = -1, dampratio: typing.SupportsFloat | typing.SupportsIndex = -1, ctrlspec: typing.SupportsInt | typing.SupportsIndex = 0) -> None"""
    def set_to_pid(self, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = ..., dampratio: typing.SupportsFloat | typing.SupportsIndex = ..., ki: typing.SupportsFloat | typing.SupportsIndex = ..., imax: typing.SupportsFloat | typing.SupportsIndex = ..., slewmax: typing.SupportsFloat | typing.SupportsIndex = ..., inheritrange: typing.SupportsFloat | typing.SupportsIndex = ..., ctrlspec: typing.SupportsInt | typing.SupportsIndex = ...) -> None:
        """set_to_pid(self: mujoco._specs.MjsActuator, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = -1, dampratio: typing.SupportsFloat | typing.SupportsIndex = -1, ki: typing.SupportsFloat | typing.SupportsIndex = -1, imax: typing.SupportsFloat | typing.SupportsIndex = -1, slewmax: typing.SupportsFloat | typing.SupportsIndex = -1, inheritrange: typing.SupportsFloat | typing.SupportsIndex = 0, ctrlspec: typing.SupportsInt | typing.SupportsIndex = 0) -> None"""
    def set_to_position(self, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = ..., dampratio: typing.SupportsFloat | typing.SupportsIndex = ..., timeconst: typing.SupportsFloat | typing.SupportsIndex = ..., inheritrange: bool = ...) -> None:
        """set_to_position(self: mujoco._specs.MjsActuator, kp: typing.SupportsFloat | typing.SupportsIndex, kv: typing.SupportsFloat | typing.SupportsIndex = -1, dampratio: typing.SupportsFloat | typing.SupportsIndex = -1, timeconst: typing.SupportsFloat | typing.SupportsIndex = -1, inheritrange: bool = False) -> None"""
    def set_to_velocity(self, kv: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """set_to_velocity(self: mujoco._specs.MjsActuator, kv: typing.SupportsFloat | typing.SupportsIndex) -> None"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsActuator) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsActuator) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsActuator) -> int"""

class MjsAuthored:
    disableactuator: int
    disableflags: int
    enableflags: int
    option: int
    visual_global: int
    visual_headlight: int
    visual_map: int
    visual_quality: int
    visual_rgba: int
    visual_scale: int
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjsBody:
    alt: MjsOrientation
    childclass: str
    classname: MjsDefault
    explicitinertial: bool
    fullinertia: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]', 'flags.writeable']
    fuse: int
    gravcomp: float
    ialt: MjsOrientation
    inertia: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    info: str
    ipos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    iquat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    mass: float
    mocap: bool
    name: str
    plugin: MjsPlugin
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    simple: int
    sleep: mujoco._enums.mjtSleepPolicy
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def add_body(self, default: MjsDefault = ..., name: str | None = ..., childclass: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mass: typing.SupportsFloat | typing.SupportsIndex | None = ..., ipos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., iquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., inertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., iaxisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., ixyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., izaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., ieuler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., fullinertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mocap: typing.SupportsInt | typing.SupportsIndex | None = ..., gravcomp: typing.SupportsFloat | typing.SupportsIndex | None = ..., sleep: typing.SupportsInt | typing.SupportsIndex | None = ..., simple: typing.SupportsInt | typing.SupportsIndex | None = ..., fuse: typing.SupportsInt | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., explicitinertial: typing.SupportsInt | typing.SupportsIndex | None = ..., plugin: MjsPlugin | None = ..., info: str | None = ...) -> MjsBody:
        """add_body(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, name: str | None = None, childclass: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mass: typing.SupportsFloat | typing.SupportsIndex | None = None, ipos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, iquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, iaxisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ixyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, izaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ieuler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fullinertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mocap: typing.SupportsInt | typing.SupportsIndex | None = None, gravcomp: typing.SupportsFloat | typing.SupportsIndex | None = None, sleep: typing.SupportsInt | typing.SupportsIndex | None = None, simple: typing.SupportsInt | typing.SupportsIndex | None = None, fuse: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, explicitinertial: typing.SupportsInt | typing.SupportsIndex | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> mujoco._specs.MjsBody


              Add body to spec.

              Args:
                name: str
                childclass: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                mass: float
                ipos: list[float]
                iquat: list[float]
                inertia: list[float]
                iaxisangle: list[float]
                ixyaxes: list[float]
                izaxis: list[float]
                ieuler: list[float]
                fullinertia: list[float]
                mocap: int
                gravcomp: float
                sleep: int
                simple: int
                fuse: int
                userdata: list[float]
                explicitinertial: int
                plugin: MjsPlugin
                info: str
      
        """
    def add_camera(self, default: MjsDefault = ..., name: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mode: typing.SupportsInt | typing.SupportsIndex | None = ..., targetbody: str | None = ..., proj: typing.SupportsInt | typing.SupportsIndex | None = ..., resolution: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., output: typing.SupportsInt | typing.SupportsIndex | None = ..., fovy: typing.SupportsFloat | typing.SupportsIndex | None = ..., ipd: typing.SupportsFloat | typing.SupportsIndex | None = ..., intrinsic: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., sensor_size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., focal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., focal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., principal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., principal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsCamera:
        """add_camera(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mode: typing.SupportsInt | typing.SupportsIndex | None = None, targetbody: str | None = None, proj: typing.SupportsInt | typing.SupportsIndex | None = None, resolution: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, output: typing.SupportsInt | typing.SupportsIndex | None = None, fovy: typing.SupportsFloat | typing.SupportsIndex | None = None, ipd: typing.SupportsFloat | typing.SupportsIndex | None = None, intrinsic: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, sensor_size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, focal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, focal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, principal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, principal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsCamera


              Add camera to spec.

              Args:
                name: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                mode: int
                targetbody: str
                proj: int
                resolution: list[float]
                output: int
                fovy: float
                ipd: float
                intrinsic: list[float]
                sensor_size: list[float]
                focal_length: list[float]
                focal_pixel: list[float]
                principal_length: list[float]
                principal_pixel: list[float]
                userdata: list[float]
                info: str
      
        """
    def add_frame(self, default: MjsFrame = ..., name: str | None = ..., childclass: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., scale: object | None = ..., scaling: typing.SupportsInt | typing.SupportsIndex | None = ..., massscale: typing.SupportsFloat | typing.SupportsIndex | None = ..., info: str | None = ...) -> MjsFrame:
        """add_frame(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsFrame = None, name: str | None = None, childclass: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, scale: object | None = None, scaling: typing.SupportsInt | typing.SupportsIndex | None = None, massscale: typing.SupportsFloat | typing.SupportsIndex | None = None, info: str | None = None) -> mujoco._specs.MjsFrame


              Add frame to spec.

              Args:
                name: str
                childclass: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                scale: float | list[float]
                scaling: int
                massscale: float
                info: str
      
        """
    def add_freejoint(self, **kwargs) -> MjsJoint:
        """add_freejoint(self: mujoco._specs.MjsBody, **kwargs) -> mujoco._specs.MjsJoint"""
    def add_geom(self, default: MjsDefault = ..., name: str | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., size: object | None = ..., contype: typing.SupportsInt | typing.SupportsIndex | None = ..., conaffinity: typing.SupportsInt | typing.SupportsIndex | None = ..., condim: typing.SupportsInt | typing.SupportsIndex | None = ..., priority: typing.SupportsInt | typing.SupportsIndex | None = ..., friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solmix: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., margin: typing.SupportsFloat | typing.SupportsIndex | None = ..., gap: typing.SupportsFloat | typing.SupportsIndex | None = ..., surfacevel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., adhesion: typing.SupportsFloat | typing.SupportsIndex | None = ..., mass: typing.SupportsFloat | typing.SupportsIndex | None = ..., density: typing.SupportsFloat | typing.SupportsIndex | None = ..., typeinertia: typing.SupportsInt | typing.SupportsIndex | None = ..., fluid_ellipsoid: typing.SupportsInt | typing.SupportsIndex | None = ..., fluid_coefs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., material: str | None = ..., rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., castshadow: typing.SupportsInt | typing.SupportsIndex | None = ..., hfieldname: str | None = ..., meshname: str | None = ..., fitscale: typing.SupportsFloat | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., plugin: MjsPlugin | None = ..., info: str | None = ...) -> MjsGeom:
        """add_geom(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: object | None = None, contype: typing.SupportsInt | typing.SupportsIndex | None = None, conaffinity: typing.SupportsInt | typing.SupportsIndex | None = None, condim: typing.SupportsInt | typing.SupportsIndex | None = None, priority: typing.SupportsInt | typing.SupportsIndex | None = None, friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solmix: typing.SupportsFloat | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, gap: typing.SupportsFloat | typing.SupportsIndex | None = None, surfacevel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, adhesion: typing.SupportsFloat | typing.SupportsIndex | None = None, mass: typing.SupportsFloat | typing.SupportsIndex | None = None, density: typing.SupportsFloat | typing.SupportsIndex | None = None, typeinertia: typing.SupportsInt | typing.SupportsIndex | None = None, fluid_ellipsoid: typing.SupportsInt | typing.SupportsIndex | None = None, fluid_coefs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, material: str | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, castshadow: typing.SupportsInt | typing.SupportsIndex | None = None, hfieldname: str | None = None, meshname: str | None = None, fitscale: typing.SupportsFloat | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> mujoco._specs.MjsGeom


              Add geom to spec.

              Args:
                name: str
                type: int
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                fromto: list[float]
                size: Optional[list[float]]
                contype: int
                conaffinity: int
                condim: int
                priority: int
                friction: list[float]
                solmix: float
                solref: list[float]
                solimp: list[float]
                margin: float
                gap: float
                surfacevel: list[float]
                adhesion: float
                mass: float
                density: float
                typeinertia: int
                fluid_ellipsoid: int
                fluid_coefs: list[float]
                material: str
                rgba: list[float]
                group: int
                castshadow: int
                hfieldname: str
                meshname: str
                fitscale: float
                userdata: list[float]
                plugin: MjsPlugin
                info: str
      
        """
    def add_joint(self, default: MjsDefault = ..., name: str | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., ref: typing.SupportsFloat | typing.SupportsIndex | None = ..., align: typing.SupportsInt | typing.SupportsIndex | None = ..., stiffness: object | None = ..., springref: typing.SupportsFloat | typing.SupportsIndex | None = ..., springyield: typing.SupportsFloat | typing.SupportsIndex | None = ..., springdamper: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., limited: typing.SupportsInt | typing.SupportsIndex | None = ..., range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., margin: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = ..., actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., armature: typing.SupportsFloat | typing.SupportsIndex | None = ..., damping: object | None = ..., frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., actgravcomp: typing.SupportsInt | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsJoint:
        """add_joint(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ref: typing.SupportsFloat | typing.SupportsIndex | None = None, align: typing.SupportsInt | typing.SupportsIndex | None = None, stiffness: object | None = None, springref: typing.SupportsFloat | typing.SupportsIndex | None = None, springyield: typing.SupportsFloat | typing.SupportsIndex | None = None, springdamper: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, limited: typing.SupportsInt | typing.SupportsIndex | None = None, range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = None, actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, armature: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: object | None = None, frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, actgravcomp: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsJoint


              Add joint to spec.

              Args:
                name: str
                type: int
                pos: list[float]
                axis: list[float]
                ref: float
                align: int
                stiffness: Optional[list[float]]
                springref: float
                springyield: float
                springdamper: list[float]
                limited: int
                range: list[float]
                margin: float
                solref_limit: list[float]
                solimp_limit: list[float]
                actfrclimited: int
                actfrcrange: list[float]
                armature: float
                damping: Optional[list[float]]
                frictionloss: float
                solref_friction: list[float]
                solimp_friction: list[float]
                group: int
                actgravcomp: int
                userdata: list[float]
                info: str
      
        """
    def add_light(self, default: MjsDefault = ..., name: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., dir: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mode: typing.SupportsInt | typing.SupportsIndex | None = ..., targetbody: str | None = ..., active: typing.SupportsInt | typing.SupportsIndex | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., texture: str | None = ..., castshadow: typing.SupportsInt | typing.SupportsIndex | None = ..., bulbradius: typing.SupportsFloat | typing.SupportsIndex | None = ..., intensity: typing.SupportsFloat | typing.SupportsIndex | None = ..., range: typing.SupportsFloat | typing.SupportsIndex | None = ..., attenuation: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., cutoff: typing.SupportsFloat | typing.SupportsIndex | None = ..., softness: typing.SupportsFloat | typing.SupportsIndex | None = ..., exponent: typing.SupportsFloat | typing.SupportsIndex | None = ..., ambient: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., diffuse: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., specular: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsLight:
        """add_light(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, dir: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mode: typing.SupportsInt | typing.SupportsIndex | None = None, targetbody: str | None = None, active: typing.SupportsInt | typing.SupportsIndex | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, texture: str | None = None, castshadow: typing.SupportsInt | typing.SupportsIndex | None = None, bulbradius: typing.SupportsFloat | typing.SupportsIndex | None = None, intensity: typing.SupportsFloat | typing.SupportsIndex | None = None, range: typing.SupportsFloat | typing.SupportsIndex | None = None, attenuation: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, cutoff: typing.SupportsFloat | typing.SupportsIndex | None = None, softness: typing.SupportsFloat | typing.SupportsIndex | None = None, exponent: typing.SupportsFloat | typing.SupportsIndex | None = None, ambient: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, diffuse: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, specular: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsLight


              Add light to spec.

              Args:
                name: str
                pos: list[float]
                dir: list[float]
                mode: int
                targetbody: str
                active: int
                type: int
                texture: str
                castshadow: int
                bulbradius: float
                intensity: float
                range: float
                attenuation: list[float]
                cutoff: float
                softness: float
                exponent: float
                ambient: list[float]
                diffuse: list[float]
                specular: list[float]
                info: str
      
        """
    def add_site(self, default: MjsDefault = ..., name: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., size: object | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., material: str | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., meshname: str | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsSite:
        """add_site(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: object | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, material: str | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, meshname: str | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsSite


              Add site to spec.

              Args:
                name: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                fromto: list[float]
                size: Optional[list[float]]
                type: int
                material: str
                group: int
                rgba: list[float]
                meshname: str
                userdata: list[float]
                info: str
      
        """
    def attach_frame(self, frame: MjsFrame, prefix: str | None = ..., suffix: str | None = ...) -> MjsFrame:
        """attach_frame(self: mujoco._specs.MjsBody, frame: mujoco._specs.MjsFrame, prefix: str | None = None, suffix: str | None = None) -> mujoco._specs.MjsFrame"""
    @overload
    def find_all(self, arg0: mujoco._enums.mjtObj) -> list:
        """find_all(*args, **kwargs)
        Overloaded function.

        1. find_all(self: mujoco._specs.MjsBody, arg0: mujoco._enums.mjtObj) -> list

        2. find_all(self: mujoco._specs.MjsBody, arg0: str) -> list
        """
    @overload
    def find_all(self, arg0: str) -> list:
        """find_all(*args, **kwargs)
        Overloaded function.

        1. find_all(self: mujoco._specs.MjsBody, arg0: mujoco._enums.mjtObj) -> list

        2. find_all(self: mujoco._specs.MjsBody, arg0: str) -> list
        """
    def find_child(self, arg0: str) -> MjsBody:
        """find_child(self: mujoco._specs.MjsBody, arg0: str) -> mujoco._specs.MjsBody"""
    def first_body(self) -> MjsBody:
        """first_body(self: mujoco._specs.MjsBody) -> mujoco._specs.MjsBody"""
    def first_camera(self) -> MjsCamera:
        """first_camera(self: mujoco._specs.MjsBody) -> mujoco._specs.MjsCamera"""
    def first_frame(self) -> MjsFrame:
        """first_frame(self: mujoco._specs.MjsBody) -> mujoco._specs.MjsFrame"""
    def first_geom(self) -> MjsGeom:
        """first_geom(self: mujoco._specs.MjsBody) -> mujoco._specs.MjsGeom"""
    def first_joint(self) -> MjsJoint:
        """first_joint(self: mujoco._specs.MjsBody) -> mujoco._specs.MjsJoint"""
    def first_light(self) -> MjsLight:
        """first_light(self: mujoco._specs.MjsBody) -> mujoco._specs.MjsLight"""
    def first_site(self) -> MjsSite:
        """first_site(self: mujoco._specs.MjsBody) -> mujoco._specs.MjsSite"""
    def make_flex(self, name: str, type: str | None = ..., dim: typing.SupportsInt | typing.SupportsIndex = ..., dof: str | None = ..., count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., cellcount: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = ..., spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., scale: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., radius: typing.SupportsFloat | typing.SupportsIndex = ..., mass: typing.SupportsFloat | typing.SupportsIndex = ..., inertiabox: typing.SupportsFloat | typing.SupportsIndex = ..., equality: typing.SupportsInt | typing.SupportsIndex = ..., rigid: typing.SupportsInt | typing.SupportsIndex = ..., flatskin: typing.SupportsInt | typing.SupportsIndex = ..., elastic2d: typing.SupportsInt | typing.SupportsIndex = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., origin: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., file: str | None = ..., mesh: str | None = ..., vfs: MjVfs = ...) -> MjsFlex:
        """make_flex(self: mujoco._specs.MjsBody, name: str, type: str | None = None, dim: typing.SupportsInt | typing.SupportsIndex = 3, dof: str | None = None, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, cellcount: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] | None = None, spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, scale: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, radius: typing.SupportsFloat | typing.SupportsIndex = 0.0, mass: typing.SupportsFloat | typing.SupportsIndex = 1.0, inertiabox: typing.SupportsFloat | typing.SupportsIndex = 0.005, equality: typing.SupportsInt | typing.SupportsIndex = 0, rigid: typing.SupportsInt | typing.SupportsIndex = 0, flatskin: typing.SupportsInt | typing.SupportsIndex = 0, elastic2d: typing.SupportsInt | typing.SupportsIndex = 0, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, origin: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, file: str | None = None, mesh: str | None = None, vfs: mujoco._specs.MjVfs = None) -> mujoco._specs.MjsFlex"""
    def next_body(self, arg0: MjsBody) -> MjsBody:
        """next_body(self: mujoco._specs.MjsBody, arg0: mujoco._specs.MjsBody) -> mujoco._specs.MjsBody"""
    def next_camera(self, arg0: MjsCamera) -> MjsCamera:
        """next_camera(self: mujoco._specs.MjsBody, arg0: mujoco._specs.MjsCamera) -> mujoco._specs.MjsCamera"""
    def next_frame(self, arg0: MjsFrame) -> MjsFrame:
        """next_frame(self: mujoco._specs.MjsBody, arg0: mujoco._specs.MjsFrame) -> mujoco._specs.MjsFrame"""
    def next_geom(self, arg0: MjsGeom) -> MjsGeom:
        """next_geom(self: mujoco._specs.MjsBody, arg0: mujoco._specs.MjsGeom) -> mujoco._specs.MjsGeom"""
    def next_joint(self, arg0: MjsJoint) -> MjsJoint:
        """next_joint(self: mujoco._specs.MjsBody, arg0: mujoco._specs.MjsJoint) -> mujoco._specs.MjsJoint"""
    def next_light(self, arg0: MjsLight) -> MjsLight:
        """next_light(self: mujoco._specs.MjsBody, arg0: mujoco._specs.MjsLight) -> mujoco._specs.MjsLight"""
    def next_site(self, arg0: MjsSite) -> MjsSite:
        """next_site(self: mujoco._specs.MjsBody, arg0: mujoco._specs.MjsSite) -> mujoco._specs.MjsSite"""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsBody, arg0: mujoco._specs.MjsFrame) -> None"""
    def to_frame(self) -> MjsFrame:
        """to_frame(self: mujoco._specs.MjsBody) -> mujoco._specs.MjsFrame"""
    @property
    def bodies(self) -> list:
        """(arg0: mujoco._specs.MjsBody) -> list"""
    @property
    def cameras(self) -> list:
        """(arg0: mujoco._specs.MjsBody) -> list"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsBody) -> mujoco._specs.MjsCompiler"""
    @property
    def frame(self) -> MjsFrame:
        """(arg0: mujoco._specs.MjsBody) -> mujoco._specs.MjsFrame"""
    @property
    def frames(self) -> list:
        """(arg0: mujoco._specs.MjsBody) -> list"""
    @property
    def geoms(self) -> list:
        """(arg0: mujoco._specs.MjsBody) -> list"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsBody) -> int"""
    @property
    def joints(self) -> list:
        """(arg0: mujoco._specs.MjsBody) -> list"""
    @property
    def lights(self) -> list:
        """(arg0: mujoco._specs.MjsBody) -> list"""
    @property
    def parent(self) -> MjsBody:
        """(arg0: mujoco._specs.MjsBody) -> mujoco._specs.MjsBody"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsBody) -> int"""
    @property
    def sites(self) -> list:
        """(arg0: mujoco._specs.MjsBody) -> list"""

class MjsCamera:
    alt: MjsOrientation
    classname: MjsDefault
    focal_length: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    focal_pixel: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    fovy: float
    info: str
    intrinsic: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    ipd: float
    mode: mujoco._enums.mjtCamLight
    name: str
    output: int
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    principal_length: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    principal_pixel: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    proj: mujoco._enums.mjtProjection
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    resolution: typing.Annotated[numpy.typing.NDArray[numpy.int32], '[2, 1]', 'flags.writeable']
    sensor_size: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    targetbody: str
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsCamera, arg0: mujoco._specs.MjsFrame) -> None"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsCamera) -> mujoco._specs.MjsCompiler"""
    @property
    def frame(self) -> MjsFrame:
        """(arg0: mujoco._specs.MjsCamera) -> mujoco._specs.MjsFrame"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsCamera) -> int"""
    @property
    def parent(self) -> MjsBody:
        """(arg0: mujoco._specs.MjsCamera) -> mujoco._specs.MjsBody"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsCamera) -> int"""

class MjsCompiler:
    LRopt: mujoco._structs.MjLROpt
    alignfree: bool
    authored: int
    autolimits: bool
    balanceinertia: bool
    boundinertia: float
    boundmass: float
    conflict: mujoco._enums.mjtConflict
    degree: bool
    discardvisual: bool
    eulerseq: MjCharVec
    fitaabb: bool
    fusestatic: bool
    inertiafromgeom: mujoco._enums.mjtInertiaFromGeom
    inertiagrouprange: typing.Annotated[numpy.typing.NDArray[numpy.int32], '[2, 1]', 'flags.writeable']
    meshdir: str
    passthrough: mujoco._enums.mjtPassthrough
    savecanonical: bool
    savecompiled: bool
    saveinertial: bool
    settotalmass: float
    texturedir: str
    usethread: bool
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjsDefault:
    actuator: MjsActuator
    camera: MjsCamera
    equality: MjsEquality
    flex: MjsFlex
    geom: MjsGeom
    joint: MjsJoint
    light: MjsLight
    material: MjsMaterial
    mesh: MjsMesh
    name: str
    pair: MjsPair
    site: MjsSite
    tendon: MjsTendon
    texture: MjsTexture
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjsElement:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjsEquality:
    active: bool
    classname: MjsDefault
    data: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[12, 1]', 'flags.writeable']
    info: str
    name: str
    name1: str
    name2: str
    objtype: mujoco._enums.mjtObj
    solimp: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solref: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    type: mujoco._enums.mjtEq
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsEquality) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsEquality) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsEquality) -> int"""

class MjsExclude:
    bodyname1: str
    bodyname2: str
    info: str
    name: str
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsExclude) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsExclude) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsExclude) -> int"""

class MjsFlex:
    activelayers: int
    cellcount: typing.Annotated[numpy.typing.NDArray[numpy.int32], '[3, 1]', 'flags.writeable']
    conaffinity: int
    condim: int
    contype: int
    damping: float
    dim: int
    edgedamping: float
    edgestiffness: float
    elastic2d: int
    elastic3d: int
    elem: MjIntVec
    elemtexcoord: MjIntVec
    flatskin: bool
    friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    gap: float
    group: int
    info: str
    margin: float
    material: MjStringVec
    name: str
    node: MjDoubleVec
    nodebody: MjStringVec
    order: int
    passive: int
    poisson: float
    priority: int
    radius: float
    rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    section: mujoco._enums.mjtFlexSection
    sectionsize: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    selfcollide: mujoco._enums.mjtFlexSelf
    shear: float
    size: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    solimp: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solmix: float
    solref: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    stressyield: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    texcoord: MjFloatVec
    thickness: float
    vert: MjDoubleVec
    vertbody: MjStringVec
    young: float
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsFlex) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsFlex) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsFlex) -> int"""

class MjsFrame:
    alt: MjsOrientation
    childclass: str
    info: str
    massscale: float
    name: str
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    scale: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    scaling: mujoco._enums.mjtScaling
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def add_body(self, default: MjsDefault = ..., name: str | None = ..., childclass: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mass: typing.SupportsFloat | typing.SupportsIndex | None = ..., ipos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., iquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., inertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., iaxisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., ixyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., izaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., ieuler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., fullinertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mocap: typing.SupportsInt | typing.SupportsIndex | None = ..., gravcomp: typing.SupportsFloat | typing.SupportsIndex | None = ..., sleep: typing.SupportsInt | typing.SupportsIndex | None = ..., simple: typing.SupportsInt | typing.SupportsIndex | None = ..., fuse: typing.SupportsInt | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., explicitinertial: typing.SupportsInt | typing.SupportsIndex | None = ..., plugin: MjsPlugin | None = ..., info: str | None = ...) -> MjsBody:
        """add_body(self: mujoco._specs.MjsFrame, default: mujoco._specs.MjsDefault = None, name: str | None = None, childclass: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mass: typing.SupportsFloat | typing.SupportsIndex | None = None, ipos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, iquat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, inertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, iaxisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ixyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, izaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ieuler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fullinertia: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mocap: typing.SupportsInt | typing.SupportsIndex | None = None, gravcomp: typing.SupportsFloat | typing.SupportsIndex | None = None, sleep: typing.SupportsInt | typing.SupportsIndex | None = None, simple: typing.SupportsInt | typing.SupportsIndex | None = None, fuse: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, explicitinertial: typing.SupportsInt | typing.SupportsIndex | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> mujoco._specs.MjsBody


              Add body to spec.

              Args:
                name: str
                childclass: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                mass: float
                ipos: list[float]
                iquat: list[float]
                inertia: list[float]
                iaxisangle: list[float]
                ixyaxes: list[float]
                izaxis: list[float]
                ieuler: list[float]
                fullinertia: list[float]
                mocap: int
                gravcomp: float
                sleep: int
                simple: int
                fuse: int
                userdata: list[float]
                explicitinertial: int
                plugin: MjsPlugin
                info: str
      
        """
    def add_camera(self, default: MjsDefault = ..., name: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mode: typing.SupportsInt | typing.SupportsIndex | None = ..., targetbody: str | None = ..., proj: typing.SupportsInt | typing.SupportsIndex | None = ..., resolution: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., output: typing.SupportsInt | typing.SupportsIndex | None = ..., fovy: typing.SupportsFloat | typing.SupportsIndex | None = ..., ipd: typing.SupportsFloat | typing.SupportsIndex | None = ..., intrinsic: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., sensor_size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., focal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., focal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., principal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., principal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsCamera:
        """add_camera(self: mujoco._specs.MjsFrame, default: mujoco._specs.MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mode: typing.SupportsInt | typing.SupportsIndex | None = None, targetbody: str | None = None, proj: typing.SupportsInt | typing.SupportsIndex | None = None, resolution: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, output: typing.SupportsInt | typing.SupportsIndex | None = None, fovy: typing.SupportsFloat | typing.SupportsIndex | None = None, ipd: typing.SupportsFloat | typing.SupportsIndex | None = None, intrinsic: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, sensor_size: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, focal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, focal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, principal_length: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, principal_pixel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsCamera


              Add camera to spec.

              Args:
                name: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                mode: int
                targetbody: str
                proj: int
                resolution: list[float]
                output: int
                fovy: float
                ipd: float
                intrinsic: list[float]
                sensor_size: list[float]
                focal_length: list[float]
                focal_pixel: list[float]
                principal_length: list[float]
                principal_pixel: list[float]
                userdata: list[float]
                info: str
      
        """
    def add_frame(self, default: MjsFrame = ..., name: str | None = ..., childclass: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., scale: object | None = ..., scaling: typing.SupportsInt | typing.SupportsIndex | None = ..., massscale: typing.SupportsFloat | typing.SupportsIndex | None = ..., info: str | None = ...) -> MjsFrame:
        """add_frame(self: mujoco._specs.MjsFrame, default: mujoco._specs.MjsFrame = None, name: str | None = None, childclass: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, scale: object | None = None, scaling: typing.SupportsInt | typing.SupportsIndex | None = None, massscale: typing.SupportsFloat | typing.SupportsIndex | None = None, info: str | None = None) -> mujoco._specs.MjsFrame


              Add frame to spec.

              Args:
                name: str
                childclass: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                scale: float | list[float]
                scaling: int
                massscale: float
                info: str
      
        """
    def add_geom(self, default: MjsDefault = ..., name: str | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., size: object | None = ..., contype: typing.SupportsInt | typing.SupportsIndex | None = ..., conaffinity: typing.SupportsInt | typing.SupportsIndex | None = ..., condim: typing.SupportsInt | typing.SupportsIndex | None = ..., priority: typing.SupportsInt | typing.SupportsIndex | None = ..., friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solmix: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., margin: typing.SupportsFloat | typing.SupportsIndex | None = ..., gap: typing.SupportsFloat | typing.SupportsIndex | None = ..., surfacevel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., adhesion: typing.SupportsFloat | typing.SupportsIndex | None = ..., mass: typing.SupportsFloat | typing.SupportsIndex | None = ..., density: typing.SupportsFloat | typing.SupportsIndex | None = ..., typeinertia: typing.SupportsInt | typing.SupportsIndex | None = ..., fluid_ellipsoid: typing.SupportsInt | typing.SupportsIndex | None = ..., fluid_coefs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., material: str | None = ..., rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., castshadow: typing.SupportsInt | typing.SupportsIndex | None = ..., hfieldname: str | None = ..., meshname: str | None = ..., fitscale: typing.SupportsFloat | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., plugin: MjsPlugin | None = ..., info: str | None = ...) -> MjsGeom:
        """add_geom(self: mujoco._specs.MjsFrame, default: mujoco._specs.MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: object | None = None, contype: typing.SupportsInt | typing.SupportsIndex | None = None, conaffinity: typing.SupportsInt | typing.SupportsIndex | None = None, condim: typing.SupportsInt | typing.SupportsIndex | None = None, priority: typing.SupportsInt | typing.SupportsIndex | None = None, friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solmix: typing.SupportsFloat | typing.SupportsIndex | None = None, solref: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, gap: typing.SupportsFloat | typing.SupportsIndex | None = None, surfacevel: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, adhesion: typing.SupportsFloat | typing.SupportsIndex | None = None, mass: typing.SupportsFloat | typing.SupportsIndex | None = None, density: typing.SupportsFloat | typing.SupportsIndex | None = None, typeinertia: typing.SupportsInt | typing.SupportsIndex | None = None, fluid_ellipsoid: typing.SupportsInt | typing.SupportsIndex | None = None, fluid_coefs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, material: str | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, castshadow: typing.SupportsInt | typing.SupportsIndex | None = None, hfieldname: str | None = None, meshname: str | None = None, fitscale: typing.SupportsFloat | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, plugin: mujoco._specs.MjsPlugin | None = None, info: str | None = None) -> mujoco._specs.MjsGeom


              Add geom to spec.

              Args:
                name: str
                type: int
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                fromto: list[float]
                size: Optional[list[float]]
                contype: int
                conaffinity: int
                condim: int
                priority: int
                friction: list[float]
                solmix: float
                solref: list[float]
                solimp: list[float]
                margin: float
                gap: float
                surfacevel: list[float]
                adhesion: float
                mass: float
                density: float
                typeinertia: int
                fluid_ellipsoid: int
                fluid_coefs: list[float]
                material: str
                rgba: list[float]
                group: int
                castshadow: int
                hfieldname: str
                meshname: str
                fitscale: float
                userdata: list[float]
                plugin: MjsPlugin
                info: str
      
        """
    def add_joint(self, default: MjsDefault = ..., name: str | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., ref: typing.SupportsFloat | typing.SupportsIndex | None = ..., align: typing.SupportsInt | typing.SupportsIndex | None = ..., stiffness: object | None = ..., springref: typing.SupportsFloat | typing.SupportsIndex | None = ..., springyield: typing.SupportsFloat | typing.SupportsIndex | None = ..., springdamper: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., limited: typing.SupportsInt | typing.SupportsIndex | None = ..., range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., margin: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = ..., actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., armature: typing.SupportsFloat | typing.SupportsIndex | None = ..., damping: object | None = ..., frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = ..., solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., actgravcomp: typing.SupportsInt | typing.SupportsIndex | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsJoint:
        """add_joint(self: mujoco._specs.MjsFrame, default: mujoco._specs.MjsDefault = None, name: str | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, ref: typing.SupportsFloat | typing.SupportsIndex | None = None, align: typing.SupportsInt | typing.SupportsIndex | None = None, stiffness: object | None = None, springref: typing.SupportsFloat | typing.SupportsIndex | None = None, springyield: typing.SupportsFloat | typing.SupportsIndex | None = None, springdamper: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, limited: typing.SupportsInt | typing.SupportsIndex | None = None, range: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, margin: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_limit: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, actfrclimited: typing.SupportsInt | typing.SupportsIndex | None = None, actfrcrange: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, armature: typing.SupportsFloat | typing.SupportsIndex | None = None, damping: object | None = None, frictionloss: typing.SupportsFloat | typing.SupportsIndex | None = None, solref_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, solimp_friction: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, actgravcomp: typing.SupportsInt | typing.SupportsIndex | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsJoint


              Add joint to spec.

              Args:
                name: str
                type: int
                pos: list[float]
                axis: list[float]
                ref: float
                align: int
                stiffness: Optional[list[float]]
                springref: float
                springyield: float
                springdamper: list[float]
                limited: int
                range: list[float]
                margin: float
                solref_limit: list[float]
                solimp_limit: list[float]
                actfrclimited: int
                actfrcrange: list[float]
                armature: float
                damping: Optional[list[float]]
                frictionloss: float
                solref_friction: list[float]
                solimp_friction: list[float]
                group: int
                actgravcomp: int
                userdata: list[float]
                info: str
      
        """
    def add_light(self, default: MjsDefault = ..., name: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., dir: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., mode: typing.SupportsInt | typing.SupportsIndex | None = ..., targetbody: str | None = ..., active: typing.SupportsInt | typing.SupportsIndex | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., texture: str | None = ..., castshadow: typing.SupportsInt | typing.SupportsIndex | None = ..., bulbradius: typing.SupportsFloat | typing.SupportsIndex | None = ..., intensity: typing.SupportsFloat | typing.SupportsIndex | None = ..., range: typing.SupportsFloat | typing.SupportsIndex | None = ..., attenuation: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., cutoff: typing.SupportsFloat | typing.SupportsIndex | None = ..., softness: typing.SupportsFloat | typing.SupportsIndex | None = ..., exponent: typing.SupportsFloat | typing.SupportsIndex | None = ..., ambient: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., diffuse: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., specular: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsLight:
        """add_light(self: mujoco._specs.MjsFrame, default: mujoco._specs.MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, dir: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, mode: typing.SupportsInt | typing.SupportsIndex | None = None, targetbody: str | None = None, active: typing.SupportsInt | typing.SupportsIndex | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, texture: str | None = None, castshadow: typing.SupportsInt | typing.SupportsIndex | None = None, bulbradius: typing.SupportsFloat | typing.SupportsIndex | None = None, intensity: typing.SupportsFloat | typing.SupportsIndex | None = None, range: typing.SupportsFloat | typing.SupportsIndex | None = None, attenuation: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, cutoff: typing.SupportsFloat | typing.SupportsIndex | None = None, softness: typing.SupportsFloat | typing.SupportsIndex | None = None, exponent: typing.SupportsFloat | typing.SupportsIndex | None = None, ambient: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, diffuse: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, specular: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsLight


              Add light to spec.

              Args:
                name: str
                pos: list[float]
                dir: list[float]
                mode: int
                targetbody: str
                active: int
                type: int
                texture: str
                castshadow: int
                bulbradius: float
                intensity: float
                range: float
                attenuation: list[float]
                cutoff: float
                softness: float
                exponent: float
                ambient: list[float]
                diffuse: list[float]
                specular: list[float]
                info: str
      
        """
    def add_site(self, default: MjsDefault = ..., name: str | None = ..., pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., size: object | None = ..., type: typing.SupportsInt | typing.SupportsIndex | None = ..., material: str | None = ..., group: typing.SupportsInt | typing.SupportsIndex | None = ..., rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., meshname: str | None = ..., userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = ..., info: str | None = ...) -> MjsSite:
        """add_site(self: mujoco._specs.MjsFrame, default: mujoco._specs.MjsDefault = None, name: str | None = None, pos: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, quat: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, axisangle: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, xyaxes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, zaxis: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, euler: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, fromto: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, size: object | None = None, type: typing.SupportsInt | typing.SupportsIndex | None = None, material: str | None = None, group: typing.SupportsInt | typing.SupportsIndex | None = None, rgba: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, meshname: str | None = None, userdata: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None = None, info: str | None = None) -> mujoco._specs.MjsSite


              Add site to spec.

              Args:
                name: str
                pos: list[float]
                quat: list[float]
                axisangle: list[float]
                xyaxes: list[float]
                zaxis: list[float]
                euler: list[float]
                fromto: list[float]
                size: Optional[list[float]]
                type: int
                material: str
                group: int
                rgba: list[float]
                meshname: str
                userdata: list[float]
                info: str
      
        """
    def attach_body(self, body: MjsBody, prefix: str | None = ..., suffix: str | None = ...) -> MjsBody:
        """attach_body(self: mujoco._specs.MjsFrame, body: mujoco._specs.MjsBody, prefix: str | None = None, suffix: str | None = None) -> mujoco._specs.MjsBody"""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsFrame, arg0: mujoco._specs.MjsFrame) -> None"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsFrame) -> mujoco._specs.MjsCompiler"""
    @property
    def frame(self) -> MjsFrame:
        """(arg0: mujoco._specs.MjsFrame) -> mujoco._specs.MjsFrame"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsFrame) -> int"""
    @property
    def parent(self) -> MjsBody:
        """(arg0: mujoco._specs.MjsFrame) -> mujoco._specs.MjsBody"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsFrame) -> int"""

class MjsGeom:
    adhesion: float
    alt: MjsOrientation
    castshadow: bool
    classname: MjsDefault
    conaffinity: int
    condim: int
    contype: int
    density: float
    fitscale: float
    fluid_coefs: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    fluid_ellipsoid: float
    friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    fromto: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]', 'flags.writeable']
    gap: float
    group: int
    hfieldname: str
    info: str
    margin: float
    mass: float
    material: str
    meshname: str
    name: str
    plugin: MjsPlugin
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    priority: int
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    size: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    solimp: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solmix: float
    solref: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    surfacevel: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]', 'flags.writeable']
    type: mujoco._enums.mjtGeom
    typeinertia: mujoco._enums.mjtGeomInertia
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsGeom, arg0: mujoco._specs.MjsFrame) -> None"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsGeom) -> mujoco._specs.MjsCompiler"""
    @property
    def frame(self) -> MjsFrame:
        """(arg0: mujoco._specs.MjsGeom) -> mujoco._specs.MjsFrame"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsGeom) -> int"""
    @property
    def parent(self) -> MjsBody:
        """(arg0: mujoco._specs.MjsGeom) -> mujoco._specs.MjsBody"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsGeom) -> int"""

class MjsHField:
    content_type: str
    file: str
    info: str
    name: str
    ncol: int
    nrow: int
    size: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    userdata: MjFloatVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsHField) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsHField) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsHField) -> int"""

class MjsJoint:
    actfrclimited: mujoco._enums.mjtLimited
    actfrcrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    actgravcomp: bool
    align: mujoco._enums.mjtAlignFree
    armature: float
    axis: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    classname: MjsDefault
    damping: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    frictionloss: float
    group: int
    info: str
    limited: mujoco._enums.mjtLimited
    margin: float
    name: str
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    range: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    ref: float
    solimp_friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solimp_limit: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solref_friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    solref_limit: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    springdamper: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    springref: float
    springyield: float
    stiffness: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    type: mujoco._enums.mjtJoint
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsJoint, arg0: mujoco._specs.MjsFrame) -> None"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsJoint) -> mujoco._specs.MjsCompiler"""
    @property
    def frame(self) -> MjsFrame:
        """(arg0: mujoco._specs.MjsJoint) -> mujoco._specs.MjsFrame"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsJoint) -> int"""
    @property
    def parent(self) -> MjsBody:
        """(arg0: mujoco._specs.MjsJoint) -> mujoco._specs.MjsBody"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsJoint) -> int"""

class MjsKey:
    act: MjDoubleVec
    ctrl: MjDoubleVec
    info: str
    mpos: MjDoubleVec
    mquat: MjDoubleVec
    name: str
    plastic: MjDoubleVec
    qpos: MjDoubleVec
    qvel: MjDoubleVec
    time: float
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsKey) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsKey) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsKey) -> int"""

class MjsLight:
    active: bool
    ambient: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    attenuation: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    bulbradius: float
    castshadow: bool
    classname: MjsDefault
    cutoff: float
    diffuse: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    dir: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    exponent: float
    info: str
    intensity: float
    mode: mujoco._enums.mjtCamLight
    name: str
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    range: float
    softness: float
    specular: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    targetbody: str
    texture: str
    type: mujoco._enums.mjtLightType
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsLight, arg0: mujoco._specs.MjsFrame) -> None"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsLight) -> mujoco._specs.MjsCompiler"""
    @property
    def frame(self) -> MjsFrame:
        """(arg0: mujoco._specs.MjsLight) -> mujoco._specs.MjsFrame"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsLight) -> int"""
    @property
    def parent(self) -> MjsBody:
        """(arg0: mujoco._specs.MjsLight) -> mujoco._specs.MjsBody"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsLight) -> int"""

class MjsMaterial:
    classname: MjsDefault
    emission: float
    info: str
    metallic: float
    name: str
    reflectance: float
    rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    roughness: float
    shininess: float
    specular: float
    texrepeat: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    textures: MjStringVec
    texuniform: bool
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsMaterial) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsMaterial) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsMaterial) -> int"""

class MjsMesh:
    classname: MjsDefault
    content_type: str
    file: str
    inertia: mujoco._enums.mjtMeshInertia
    info: str
    material: str
    maxhullvert: int
    name: str
    needsdf: bool
    octree_maxdepth: int
    plugin: MjsPlugin
    refpos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    refquat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    scale: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    smoothnormal: bool
    userface: MjIntVec
    userfacenormal: MjIntVec
    userfacetexcoord: MjIntVec
    usernode: MjDoubleVec
    usernormal: MjFloatVec
    usertet: MjIntVec
    usertexcoord: MjFloatVec
    uservert: MjFloatVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def make_box(self, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dim: typing.SupportsInt | typing.SupportsIndex = ...) -> None:
        """make_box(self: mujoco._specs.MjsMesh, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dim: typing.SupportsInt | typing.SupportsIndex = 3) -> None"""
    def make_cone(self, nedge: typing.SupportsInt | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """make_cone(self: mujoco._specs.MjsMesh, nedge: typing.SupportsInt | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex) -> None"""
    def make_cylinder(self, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dim: typing.SupportsInt | typing.SupportsIndex = ...) -> None:
        """make_cylinder(self: mujoco._specs.MjsMesh, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dim: typing.SupportsInt | typing.SupportsIndex = 3) -> None"""
    def make_disc(self, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        """make_disc(self: mujoco._specs.MjsMesh, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None"""
    def make_ellipsoid(self, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dim: typing.SupportsInt | typing.SupportsIndex = ...) -> None:
        """make_ellipsoid(self: mujoco._specs.MjsMesh, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dim: typing.SupportsInt | typing.SupportsIndex = 3) -> None"""
    def make_grid(self, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dim: typing.SupportsInt | typing.SupportsIndex = ...) -> None:
        """make_grid(self: mujoco._specs.MjsMesh, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], dim: typing.SupportsInt | typing.SupportsIndex = 3) -> None"""
    def make_hemisphere(self, resolution: typing.SupportsInt | typing.SupportsIndex) -> None:
        """make_hemisphere(self: mujoco._specs.MjsMesh, resolution: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def make_plate(self, resolution=...) -> None:
        '''make_plate(self: mujoco._specs.MjsMesh, resolution: typing.Annotated[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], "FixedSize(2)"] = [0, 0]) -> None'''
    def make_sphere(self, subdivision: typing.SupportsInt | typing.SupportsIndex) -> None:
        """make_sphere(self: mujoco._specs.MjsMesh, subdivision: typing.SupportsInt | typing.SupportsIndex) -> None"""
    def make_square(self, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        """make_square(self: mujoco._specs.MjsMesh, count: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], spacing: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None"""
    def make_supersphere(self, resolution: typing.SupportsInt | typing.SupportsIndex, e: typing.SupportsFloat | typing.SupportsIndex, n: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """make_supersphere(self: mujoco._specs.MjsMesh, resolution: typing.SupportsInt | typing.SupportsIndex, e: typing.SupportsFloat | typing.SupportsIndex, n: typing.SupportsFloat | typing.SupportsIndex) -> None"""
    def make_supertorus(self, resolution: typing.SupportsInt | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, s: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """make_supertorus(self: mujoco._specs.MjsMesh, resolution: typing.SupportsInt | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, s: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> None"""
    def make_wedge(self, resolution=..., fov=..., gamma: typing.SupportsFloat | typing.SupportsIndex = ...) -> None:
        '''make_wedge(self: mujoco._specs.MjsMesh, resolution: typing.Annotated[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], "FixedSize(2)"] = [0, 0], fov: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"] = [0.0, 0.0], gamma: typing.SupportsFloat | typing.SupportsIndex = 0) -> None'''
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsMesh) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsMesh) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsMesh) -> int"""

class MjsNumeric:
    data: MjDoubleVec
    info: str
    name: str
    size: int
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsNumeric) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsNumeric) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsNumeric) -> int"""

class MjsOrientation:
    axisangle: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    euler: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    type: mujoco._enums.mjtOrientation
    xyaxes: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]', 'flags.writeable']
    zaxis: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MjsPair:
    adhesion: float
    classname: MjsDefault
    condim: int
    friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    gap: float
    geomname1: str
    geomname2: str
    info: str
    margin: float
    name: str
    solimp: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solref: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    solreffriction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsPair) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsPair) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsPair) -> int"""

class MjsPlugin:
    active: bool
    config: dict
    id: int
    info: str
    name: str
    plugin_name: str
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsPlugin) -> mujoco._specs.MjsCompiler"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsPlugin) -> int"""

class MjsSensor:
    cutoff: float
    datatype: mujoco._enums.mjtDataType
    delay: float
    dim: int
    info: str
    interp: int
    interval: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    intprm: typing.Annotated[numpy.typing.NDArray[numpy.int32], '[3, 1]', 'flags.writeable']
    name: str
    needstage: mujoco._enums.mjtStage
    noise: float
    nsample: int
    objname: str
    objtype: mujoco._enums.mjtObj
    plugin: MjsPlugin
    refname: str
    reftype: mujoco._enums.mjtObj
    type: mujoco._enums.mjtSensor
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def get_data_size(self) -> int:
        """get_data_size(self: mujoco._specs.MjsSensor) -> int"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsSensor) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsSensor) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsSensor) -> int"""

class MjsSite:
    alt: MjsOrientation
    classname: MjsDefault
    fromto: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]', 'flags.writeable']
    group: int
    info: str
    material: str
    meshname: str
    name: str
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    size: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    type: mujoco._enums.mjtGeom
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def attach_body(self, body: MjsBody, prefix: str | None = ..., suffix: str | None = ...) -> MjsBody:
        """attach_body(self: mujoco._specs.MjsSite, body: mujoco._specs.MjsBody, prefix: str | None = None, suffix: str | None = None) -> mujoco._specs.MjsBody"""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsSite, arg0: mujoco._specs.MjsFrame) -> None"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsSite) -> mujoco._specs.MjsCompiler"""
    @property
    def frame(self) -> MjsFrame:
        """(arg0: mujoco._specs.MjsSite) -> mujoco._specs.MjsFrame"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsSite) -> int"""
    @property
    def parent(self) -> MjsBody:
        """(arg0: mujoco._specs.MjsSite) -> mujoco._specs.MjsBody"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsSite) -> int"""

class MjsSkin:
    bindpos: MjFloatVec
    bindquat: MjFloatVec
    bodyname: MjStringVec
    face: MjIntVec
    file: str
    group: int
    inflate: float
    info: str
    material: str
    name: str
    rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    texcoord: MjFloatVec
    vert: MjFloatVec
    vertid: list
    vertweight: list
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsSkin) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsSkin) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsSkin) -> int"""

class MjsTendon:
    actfrclimited: mujoco._enums.mjtLimited
    actfrcrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    armature: float
    damping: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    frictionloss: float
    group: int
    info: str
    limited: mujoco._enums.mjtLimited
    margin: float
    material: str
    name: str
    range: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    solimp_friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solimp_limit: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solref_friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    solref_limit: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    springlength: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    springyield: float
    stiffness: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    userdata: MjDoubleVec
    width: float
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def default(self) -> MjsDefault:
        """default(self: mujoco._specs.MjsTendon) -> mujoco._specs.MjsDefault"""
    def wrap_geom(self, arg0: str, arg1: str) -> MjsWrap:
        """wrap_geom(self: mujoco._specs.MjsTendon, arg0: str, arg1: str) -> mujoco._specs.MjsWrap"""
    def wrap_joint(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> MjsWrap:
        """wrap_joint(self: mujoco._specs.MjsTendon, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> mujoco._specs.MjsWrap"""
    def wrap_pulley(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> MjsWrap:
        """wrap_pulley(self: mujoco._specs.MjsTendon, arg0: typing.SupportsFloat | typing.SupportsIndex) -> mujoco._specs.MjsWrap"""
    def wrap_site(self, arg0: str) -> MjsWrap:
        """wrap_site(self: mujoco._specs.MjsTendon, arg0: str) -> mujoco._specs.MjsWrap"""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsTendon) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsTendon) -> int"""
    @property
    def path(self) -> MjsTendonPath:
        """(arg0: mujoco._specs.MjsTendon) -> mujoco._specs.MjsTendonPath"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsTendon) -> int"""

class MjsTendonPath:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> MjsWrap:
        """__getitem__(self: mujoco._specs.MjsTendonPath, arg0: typing.SupportsInt | typing.SupportsIndex) -> mujoco._specs.MjsWrap"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjsTendonPath) -> int"""

class MjsText:
    data: str
    info: str
    name: str
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsText) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsText) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsText) -> int"""

class MjsTexture:
    builtin: mujoco._enums.mjtBuiltin
    classname: MjsDefault
    colorspace: mujoco._enums.mjtColorSpace
    content_type: str
    cubefiles: MjStringVec
    data: bytes
    embed: mujoco._enums.mjtEmbed
    file: str
    gridlayout: MjCharVec
    gridsize: typing.Annotated[numpy.typing.NDArray[numpy.int32], '[2, 1]', 'flags.writeable']
    height: int
    hflip: bool
    info: str
    mark: mujoco._enums.mjtMark
    markrgb: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    name: str
    nchannel: int
    random: float
    rgb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    rgb2: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    type: mujoco._enums.mjtTexture
    vflip: bool
    width: int
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsTexture) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsTexture) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsTexture) -> int"""

class MjsTuple:
    info: str
    name: str
    objname: MjStringVec
    objprm: MjDoubleVec
    objtype: MjIntVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def compiler(self) -> MjsCompiler:
        """(arg0: mujoco._specs.MjsTuple) -> mujoco._specs.MjsCompiler"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsTuple) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsTuple) -> int"""

class MjsWrap:
    info: str
    type: mujoco._enums.mjtWrap
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def coef(self) -> object:
        """(arg0: mujoco._specs.MjsWrap) -> object"""
    @property
    def divisor(self) -> object:
        """(arg0: mujoco._specs.MjsWrap) -> object"""
    @property
    def sidesite(self) -> MjsSite:
        """(arg0: mujoco._specs.MjsWrap) -> mujoco._specs.MjsSite"""
    @property
    def target(self) -> object:
        """(arg0: mujoco._specs.MjsWrap) -> object"""
