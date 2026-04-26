import collections.abc
import flags
import mujoco._enums
import mujoco._structs
import numpy
import numpy.typing
import typing
from typing import Callable, ClassVar, overload

class MjByteVec:
    def __init__(self, arg0, arg1: typing.SupportsInt) -> None:
        """__init__(self: mujoco._specs.MjByteVec, arg0: std::byte, arg1: typing.SupportsInt) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __getitem__(self, index):
        """__getitem__(self: mujoco._specs.MjByteVec, arg0: typing.SupportsInt) -> std::byte"""
    def __iter__(self):
        """__iter__(self: mujoco._specs.MjByteVec) -> collections.abc.Iterator[std::byte]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjByteVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt, arg1) -> None:
        """__setitem__(self: mujoco._specs.MjByteVec, arg0: typing.SupportsInt, arg1: std::byte) -> None"""

class MjCharVec:
    def __init__(self, arg0: str, arg1: typing.SupportsInt) -> None:
        """__init__(self: mujoco._specs.MjCharVec, arg0: str, arg1: typing.SupportsInt) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __getitem__(self, arg0: typing.SupportsInt) -> str:
        """__getitem__(self: mujoco._specs.MjCharVec, arg0: typing.SupportsInt) -> str"""
    def __iter__(self) -> collections.abc.Iterator[str]:
        """__iter__(self: mujoco._specs.MjCharVec) -> collections.abc.Iterator[str]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjCharVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt, arg1: str) -> None:
        """__setitem__(self: mujoco._specs.MjCharVec, arg0: typing.SupportsInt, arg1: str) -> None"""

class MjDoubleVec:
    def __init__(self, arg0: typing.SupportsFloat, arg1: typing.SupportsInt) -> None:
        """__init__(self: mujoco._specs.MjDoubleVec, arg0: typing.SupportsFloat, arg1: typing.SupportsInt) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __getitem__(self, arg0: typing.SupportsInt) -> float:
        """__getitem__(self: mujoco._specs.MjDoubleVec, arg0: typing.SupportsInt) -> float"""
    def __iter__(self) -> collections.abc.Iterator[float]:
        """__iter__(self: mujoco._specs.MjDoubleVec) -> collections.abc.Iterator[float]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjDoubleVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None:
        """__setitem__(self: mujoco._specs.MjDoubleVec, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None"""

class MjFloatVec:
    def __init__(self, arg0: typing.SupportsFloat, arg1: typing.SupportsInt) -> None:
        """__init__(self: mujoco._specs.MjFloatVec, arg0: typing.SupportsFloat, arg1: typing.SupportsInt) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __getitem__(self, arg0: typing.SupportsInt) -> float:
        """__getitem__(self: mujoco._specs.MjFloatVec, arg0: typing.SupportsInt) -> float"""
    def __iter__(self) -> collections.abc.Iterator[float]:
        """__iter__(self: mujoco._specs.MjFloatVec) -> collections.abc.Iterator[float]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjFloatVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None:
        """__setitem__(self: mujoco._specs.MjFloatVec, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None"""

class MjIntVec:
    def __init__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        """__init__(self: mujoco._specs.MjIntVec, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        """__getitem__(self: mujoco._specs.MjIntVec, arg0: typing.SupportsInt) -> int"""
    def __iter__(self) -> collections.abc.Iterator[int]:
        """__iter__(self: mujoco._specs.MjIntVec) -> collections.abc.Iterator[int]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjIntVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        """__setitem__(self: mujoco._specs.MjIntVec, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None"""

class MjOption:
    apirate: float
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
    solver: int
    timestep: float
    tolerance: float
    viscosity: float
    wind: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

class MjSpec:
    from_zip: ClassVar[Callable] = ...
    to_zip: ClassVar[Callable] = ...
    assets: dict
    comment: str
    compiler: MjsCompiler
    copy_during_attach: None
    hasImplicitPluginElem: int
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
    strippath: int
    texturedir: str
    visual: MjVisual
    def __init__(self) -> None:
        """__init__(self: mujoco._specs.MjSpec) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def activate_plugin(self, name: str) -> None:
        """activate_plugin(self: mujoco._specs.MjSpec, name: str) -> None"""
    def actuator(self, arg0: str) -> MjsActuator:
        """actuator(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsActuator"""
    def add_actuator(self, default: MjsDefault = ..., **kwargs) -> MjsActuator:
        """add_actuator(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsActuator"""
    def add_default(self, arg0: str, arg1: MjsDefault) -> MjsDefault:
        """add_default(self: mujoco._specs.MjSpec, arg0: str, arg1: mujoco._specs.MjsDefault) -> mujoco._specs.MjsDefault"""
    def add_equality(self, default: MjsDefault = ..., **kwargs) -> MjsEquality:
        """add_equality(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsEquality"""
    def add_exclude(self, **kwargs) -> MjsExclude:
        """add_exclude(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsExclude"""
    def add_flex(self, **kwargs) -> MjsFlex:
        """add_flex(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsFlex"""
    def add_hfield(self, **kwargs) -> MjsHField:
        """add_hfield(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsHField"""
    def add_key(self, **kwargs) -> MjsKey:
        """add_key(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsKey"""
    def add_material(self, default: MjsDefault = ..., **kwargs) -> MjsMaterial:
        """add_material(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsMaterial"""
    def add_mesh(self, default: MjsDefault = ..., **kwargs) -> MjsMesh:
        """add_mesh(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsMesh"""
    def add_numeric(self, **kwargs) -> MjsNumeric:
        """add_numeric(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsNumeric"""
    def add_pair(self, default: MjsDefault = ..., **kwargs) -> MjsPair:
        """add_pair(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsPair"""
    def add_plugin(self, **kwargs) -> MjsPlugin:
        """add_plugin(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsPlugin"""
    def add_sensor(self, **kwargs) -> MjsSensor:
        """add_sensor(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsSensor"""
    def add_skin(self, **kwargs) -> MjsSkin:
        """add_skin(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsSkin"""
    def add_tendon(self, default: MjsDefault = ..., **kwargs) -> MjsTendon:
        """add_tendon(self: mujoco._specs.MjSpec, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsTendon"""
    def add_text(self, **kwargs) -> MjsText:
        """add_text(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsText"""
    def add_texture(self, **kwargs) -> MjsTexture:
        """add_texture(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsTexture"""
    def add_tuple(self, **kwargs) -> MjsTuple:
        """add_tuple(self: mujoco._specs.MjSpec, **kwargs) -> mujoco._specs.MjsTuple"""
    def attach(self, child: MjSpec, prefix: str | None = ..., suffix: str | None = ..., site: object | None = ..., frame: object | None = ...) -> MjsFrame:
        """attach(self: mujoco._specs.MjSpec, child: mujoco._specs.MjSpec, prefix: str | None = None, suffix: str | None = None, site: object | None = None, frame: object | None = None) -> mujoco._specs.MjsFrame"""
    def body(self, arg0: str) -> MjsBody:
        """body(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsBody"""
    def camera(self, arg0: str) -> MjsCamera:
        """camera(self: mujoco._specs.MjSpec, arg0: str) -> mujoco._specs.MjsCamera"""
    def compile(self) -> object:
        """compile(self: mujoco._specs.MjSpec) -> object"""
    def copy(self) -> MjSpec:
        """copy(self: mujoco._specs.MjSpec) -> mujoco._specs.MjSpec"""
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
    def from_file(filename: str, include: collections.abc.Mapping[str, bytes] | None = ..., assets: dict | None = ...) -> MjSpec:
        """from_file(filename: str, include: collections.abc.Mapping[str, bytes] | None = None, assets: dict | None = None) -> mujoco._specs.MjSpec


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
  
        """
    @staticmethod
    def from_string(xml: str, include: collections.abc.Mapping[str, bytes] | None = ..., assets: dict | None = ...) -> MjSpec:
        """from_string(xml: str, include: collections.abc.Mapping[str, bytes] | None = None, assets: dict | None = None) -> mujoco._specs.MjSpec


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
    def recompile(self, arg0: object, arg1: object) -> object:
        """recompile(self: mujoco._specs.MjSpec, arg0: object, arg1: object) -> object"""
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
    def _address(self) -> int:
        """(arg0: mujoco._specs.MjSpec) -> int"""
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

class MjStringVec:
    def __init__(self, arg0: str, arg1: typing.SupportsInt) -> None:
        """__init__(self: mujoco._specs.MjStringVec, arg0: str, arg1: typing.SupportsInt) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __getitem__(self, arg0: typing.SupportsInt) -> str:
        """__getitem__(self: mujoco._specs.MjStringVec, arg0: typing.SupportsInt) -> str"""
    def __iter__(self) -> collections.abc.Iterator[str]:
        """__iter__(self: mujoco._specs.MjStringVec) -> collections.abc.Iterator[str]"""
    def __len__(self) -> int:
        """__len__(self: mujoco._specs.MjStringVec) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt, arg1: str) -> None:
        """__setitem__(self: mujoco._specs.MjStringVec, arg0: typing.SupportsInt, arg1: str) -> None"""

class MjVisual:
    global_: mujoco._structs.MjVisual.Global
    headlight: MjVisualHeadlight
    map: mujoco._structs.MjVisual.Map
    quality: mujoco._structs.MjVisual.Quality
    rgba: MjVisualRgba
    scale: mujoco._structs.MjVisual.Scale
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

class MjVisualHeadlight:
    active: int
    ambient: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    diffuse: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    specular: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

class MjsActuator:
    actdim: int
    actearly: int
    actlimited: int
    actrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    biasprm: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[10, 1]', 'flags.writeable']
    biastype: mujoco._enums.mjtBias
    classname: MjsDefault
    cranklength: float
    ctrllimited: int
    ctrlrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    dynprm: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[10, 1]', 'flags.writeable']
    dyntype: mujoco._enums.mjtDyn
    forcelimited: int
    forcerange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    gainprm: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[10, 1]', 'flags.writeable']
    gaintype: mujoco._enums.mjtGain
    gear: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]', 'flags.writeable']
    group: int
    info: str
    inheritrange: float
    lengthrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    name: str
    plugin: MjsPlugin
    refsite: str
    slidersite: str
    target: str
    trntype: mujoco._enums.mjtTrn
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def set_to_adhesion(self, gain: typing.SupportsFloat) -> None:
        """set_to_adhesion(self: mujoco._specs.MjsActuator, gain: typing.SupportsFloat) -> None"""
    def set_to_cylinder(self, timeconst: typing.SupportsFloat, bias: typing.SupportsFloat, area: typing.SupportsFloat, diameter: typing.SupportsFloat = ...) -> None:
        """set_to_cylinder(self: mujoco._specs.MjsActuator, timeconst: typing.SupportsFloat, bias: typing.SupportsFloat, area: typing.SupportsFloat, diameter: typing.SupportsFloat = -1) -> None"""
    def set_to_damper(self, kv: typing.SupportsFloat) -> None:
        """set_to_damper(self: mujoco._specs.MjsActuator, kv: typing.SupportsFloat) -> None"""
    def set_to_intvelocity(self, kp: typing.SupportsFloat, kv: typing.SupportsFloat = ..., dampratio: typing.SupportsFloat = ..., timeconst: typing.SupportsFloat = ..., inheritrange: bool = ...) -> None:
        """set_to_intvelocity(self: mujoco._specs.MjsActuator, kp: typing.SupportsFloat, kv: typing.SupportsFloat = -1, dampratio: typing.SupportsFloat = -1, timeconst: typing.SupportsFloat = -1, inheritrange: bool = False) -> None"""
    def set_to_motor(self) -> None:
        """set_to_motor(self: mujoco._specs.MjsActuator) -> None"""
    def set_to_muscle(self, timeconst: typing.SupportsFloat = ..., tausmooth: typing.SupportsFloat, range: typing.SupportsFloat = ..., force: typing.SupportsFloat = ..., scale: typing.SupportsFloat = ..., lmin: typing.SupportsFloat = ..., lmax: typing.SupportsFloat = ..., vmax: typing.SupportsFloat = ..., fpmax: typing.SupportsFloat = ..., fvmax: typing.SupportsFloat = ...) -> None:
        """set_to_muscle(self: mujoco._specs.MjsActuator, timeconst: typing.SupportsFloat = -1, tausmooth: typing.SupportsFloat, range: typing.SupportsFloat = [-1.0, -1.0], force: typing.SupportsFloat = -1, scale: typing.SupportsFloat = -1, lmin: typing.SupportsFloat = -1, lmax: typing.SupportsFloat = -1, vmax: typing.SupportsFloat = -1, fpmax: typing.SupportsFloat = -1, fvmax: typing.SupportsFloat = -1) -> None"""
    def set_to_position(self, kp: typing.SupportsFloat, kv: typing.SupportsFloat = ..., dampratio: typing.SupportsFloat = ..., timeconst: typing.SupportsFloat = ..., inheritrange: bool = ...) -> None:
        """set_to_position(self: mujoco._specs.MjsActuator, kp: typing.SupportsFloat, kv: typing.SupportsFloat = -1, dampratio: typing.SupportsFloat = -1, timeconst: typing.SupportsFloat = -1, inheritrange: bool = False) -> None"""
    def set_to_velocity(self, kv: typing.SupportsFloat) -> None:
        """set_to_velocity(self: mujoco._specs.MjsActuator, kv: typing.SupportsFloat) -> None"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsActuator) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsActuator) -> int"""

class MjsBody:
    alt: MjsOrientation
    childclass: str
    classname: MjsDefault
    explicitinertial: int
    fullinertia: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]', 'flags.writeable']
    gravcomp: float
    ialt: MjsOrientation
    inertia: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    info: str
    ipos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    iquat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    mass: float
    mocap: int
    name: str
    plugin: MjsPlugin
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def add_body(self, default: MjsDefault = ..., **kwargs) -> MjsBody:
        """add_body(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsBody"""
    def add_camera(self, default: MjsDefault = ..., **kwargs) -> MjsCamera:
        """add_camera(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsCamera"""
    def add_frame(self, default: MjsFrame = ..., **kwargs) -> MjsFrame:
        """add_frame(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsFrame = None, **kwargs) -> mujoco._specs.MjsFrame"""
    def add_freejoint(self, **kwargs) -> MjsJoint:
        """add_freejoint(self: mujoco._specs.MjsBody, **kwargs) -> mujoco._specs.MjsJoint"""
    def add_geom(self, default: MjsDefault = ..., **kwargs) -> MjsGeom:
        """add_geom(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsGeom"""
    def add_joint(self, default: MjsDefault = ..., **kwargs) -> MjsJoint:
        """add_joint(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsJoint"""
    def add_light(self, default: MjsDefault = ..., **kwargs) -> MjsLight:
        """add_light(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsLight"""
    def add_site(self, default: MjsDefault = ..., **kwargs) -> MjsSite:
        """add_site(self: mujoco._specs.MjsBody, default: mujoco._specs.MjsDefault = None, **kwargs) -> mujoco._specs.MjsSite"""
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
    orthographic: int
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    principal_length: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    principal_pixel: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    resolution: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    sensor_size: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[2, 1]', 'flags.writeable']
    targetbody: str
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsCamera, arg0: mujoco._specs.MjsFrame) -> None"""
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
    alignfree: int
    autolimits: int
    balanceinertia: int
    boundinertia: float
    boundmass: float
    degree: int
    discardvisual: int
    eulerseq: MjCharVec
    fitaabb: int
    fusestatic: int
    inertiafromgeom: int
    inertiagrouprange: typing.Annotated[numpy.typing.NDArray[numpy.int32], '[2, 1]', 'flags.writeable']
    saveinertial: int
    settotalmass: float
    usethread: int
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

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
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

class MjsElement:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

class MjsEquality:
    active: int
    classname: MjsDefault
    data: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[11, 1]', 'flags.writeable']
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsExclude) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsExclude) -> int"""

class MjsFlex:
    activelayers: int
    conaffinity: int
    condim: int
    contype: int
    damping: float
    dim: int
    edgedamping: float
    edgestiffness: float
    elastic2d: int
    elem: MjIntVec
    elemtexcoord: MjIntVec
    flatskin: int
    friction: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    gap: float
    group: int
    info: str
    internal: int
    margin: float
    material: str
    name: str
    node: MjDoubleVec
    nodebody: MjStringVec
    passive: int
    poisson: float
    priority: int
    radius: float
    rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    selfcollide: int
    solimp: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[5, 1]', 'flags.writeable']
    solmix: float
    solref: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    texcoord: MjFloatVec
    thickness: float
    vert: MjDoubleVec
    vertbody: MjStringVec
    vertcollide: int
    young: float
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
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
    name: str
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def attach_body(self, body: MjsBody, prefix: str | None = ..., suffix: str | None = ...) -> MjsBody:
        """attach_body(self: mujoco._specs.MjsFrame, body: mujoco._specs.MjsBody, prefix: str | None = None, suffix: str | None = None) -> mujoco._specs.MjsBody"""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsFrame, arg0: mujoco._specs.MjsFrame) -> None"""
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
    alt: MjsOrientation
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
    type: mujoco._enums.mjtGeom
    typeinertia: mujoco._enums.mjtGeomInertia
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsGeom, arg0: mujoco._specs.MjsFrame) -> None"""
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsHField) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsHField) -> int"""

class MjsJoint:
    actfrclimited: int
    actfrcrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    actgravcomp: int
    align: int
    armature: float
    axis: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    classname: MjsDefault
    damping: float
    frictionloss: float
    group: int
    info: str
    limited: int
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
    stiffness: float
    type: mujoco._enums.mjtJoint
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsJoint, arg0: mujoco._specs.MjsFrame) -> None"""
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
    qpos: MjDoubleVec
    qvel: MjDoubleVec
    time: float
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsKey) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsKey) -> int"""

class MjsLight:
    active: int
    ambient: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    attenuation: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    bulbradius: float
    castshadow: int
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
    specular: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[3, 1]', 'flags.writeable']
    targetbody: str
    texture: str
    type: mujoco._enums.mjtLightType
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsLight, arg0: mujoco._specs.MjsFrame) -> None"""
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
    texuniform: int
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
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
    needsdf: int
    plugin: MjsPlugin
    refpos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    refquat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    scale: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    smoothnormal: int
    userface: MjIntVec
    userfacetexcoord: MjIntVec
    usernormal: MjFloatVec
    usertexcoord: MjFloatVec
    uservert: MjFloatVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def make_cone(self, nedge: typing.SupportsInt, radius: typing.SupportsFloat) -> None:
        """make_cone(self: mujoco._specs.MjsMesh, nedge: typing.SupportsInt, radius: typing.SupportsFloat) -> None"""
    def make_hemisphere(self, resolution: typing.SupportsInt) -> None:
        """make_hemisphere(self: mujoco._specs.MjsMesh, resolution: typing.SupportsInt) -> None"""
    def make_plate(self, resolution=...) -> None:
        '''make_plate(self: mujoco._specs.MjsMesh, resolution: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"] = [0, 0]) -> None'''
    def make_sphere(self, subdivision: typing.SupportsInt) -> None:
        """make_sphere(self: mujoco._specs.MjsMesh, subdivision: typing.SupportsInt) -> None"""
    def make_supersphere(self, resolution: typing.SupportsInt, e: typing.SupportsFloat, n: typing.SupportsFloat) -> None:
        """make_supersphere(self: mujoco._specs.MjsMesh, resolution: typing.SupportsInt, e: typing.SupportsFloat, n: typing.SupportsFloat) -> None"""
    def make_supertorus(self, resolution: typing.SupportsInt, radius: typing.SupportsFloat, s: typing.SupportsFloat, t: typing.SupportsFloat) -> None:
        """make_supertorus(self: mujoco._specs.MjsMesh, resolution: typing.SupportsInt, radius: typing.SupportsFloat, s: typing.SupportsFloat, t: typing.SupportsFloat) -> None"""
    def make_wedge(self, resolution=..., fov=..., gamma: typing.SupportsFloat = ...) -> None:
        '''make_wedge(self: mujoco._specs.MjsMesh, resolution: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"] = [0, 0], fov: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"] = [0.0, 0.0], gamma: typing.SupportsFloat = 0) -> None'''
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...

class MjsPair:
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsPair) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsPair) -> int"""

class MjsPlugin:
    active: int
    config: dict
    id: int
    info: str
    name: str
    plugin_name: str
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsPlugin) -> int"""

class MjsSensor:
    cutoff: float
    datatype: mujoco._enums.mjtDataType
    dim: int
    info: str
    intprm: typing.Annotated[numpy.typing.NDArray[numpy.int32], '[3, 1]', 'flags.writeable']
    name: str
    needstage: mujoco._enums.mjtStage
    noise: float
    objname: str
    objtype: mujoco._enums.mjtObj
    plugin: MjsPlugin
    refname: str
    reftype: mujoco._enums.mjtObj
    type: mujoco._enums.mjtSensor
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def get_data_size(self) -> int:
        """get_data_size(self: mujoco._specs.MjsSensor) -> int"""
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
    name: str
    pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[4, 1]', 'flags.writeable']
    rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], '[4, 1]', 'flags.writeable']
    size: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    type: mujoco._enums.mjtGeom
    userdata: MjDoubleVec
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def attach_body(self, body: MjsBody, prefix: str | None = ..., suffix: str | None = ...) -> MjsBody:
        """attach_body(self: mujoco._specs.MjsSite, body: mujoco._specs.MjsBody, prefix: str | None = None, suffix: str | None = None) -> mujoco._specs.MjsBody"""
    def set_frame(self, arg0: MjsFrame) -> None:
        """set_frame(self: mujoco._specs.MjsSite, arg0: mujoco._specs.MjsFrame) -> None"""
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsSkin) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsSkin) -> int"""

class MjsTendon:
    actfrclimited: int
    actfrcrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[2, 1]', 'flags.writeable']
    armature: float
    damping: float
    frictionloss: float
    group: int
    info: str
    limited: int
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
    stiffness: float
    userdata: MjDoubleVec
    width: float
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def default(self) -> MjsDefault:
        """default(self: mujoco._specs.MjsTendon) -> mujoco._specs.MjsDefault"""
    def wrap_geom(self, arg0: str, arg1: str) -> MjsWrap:
        """wrap_geom(self: mujoco._specs.MjsTendon, arg0: str, arg1: str) -> mujoco._specs.MjsWrap"""
    def wrap_joint(self, arg0: str, arg1: typing.SupportsFloat) -> MjsWrap:
        """wrap_joint(self: mujoco._specs.MjsTendon, arg0: str, arg1: typing.SupportsFloat) -> mujoco._specs.MjsWrap"""
    def wrap_pulley(self, arg0: typing.SupportsFloat) -> MjsWrap:
        """wrap_pulley(self: mujoco._specs.MjsTendon, arg0: typing.SupportsFloat) -> mujoco._specs.MjsWrap"""
    def wrap_site(self, arg0: str) -> MjsWrap:
        """wrap_site(self: mujoco._specs.MjsTendon, arg0: str) -> mujoco._specs.MjsWrap"""
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsTendon) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsTendon) -> int"""

class MjsText:
    data: str
    info: str
    name: str
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsText) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsText) -> int"""

class MjsTexture:
    builtin: int
    colorspace: mujoco._enums.mjtColorSpace
    content_type: str
    cubefiles: MjStringVec
    data: MjByteVec
    file: str
    gridlayout: MjCharVec
    gridsize: typing.Annotated[numpy.typing.NDArray[numpy.int32], '[2, 1]', 'flags.writeable']
    height: int
    hflip: int
    info: str
    mark: int
    markrgb: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    name: str
    nchannel: int
    random: float
    rgb1: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    rgb2: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]', 'flags.writeable']
    type: mujoco._enums.mjtTexture
    vflip: int
    width: int
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
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
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @property
    def id(self) -> int:
        """(arg0: mujoco._specs.MjsTuple) -> int"""
    @property
    def signature(self) -> int:
        """(arg0: mujoco._specs.MjsTuple) -> int"""

class MjsWrap:
    info: str
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
