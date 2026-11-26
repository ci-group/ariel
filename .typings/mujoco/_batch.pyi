import collections.abc
import mujoco._structs
import numpy
import numpy.typing
import typing

class Batch:
    def __init__(self, model: mujoco._structs.MjModel, nsim: typing.SupportsInt | typing.SupportsIndex, nthread: typing.SupportsInt | typing.SupportsIndex = ..., persistent: bool = ...) -> None:
        """__init__(self: mujoco._batch.Batch, model: mujoco._structs.MjModel, nsim: typing.SupportsInt | typing.SupportsIndex, nthread: typing.SupportsInt | typing.SupportsIndex = 0, persistent: bool = False) -> None"""
    def error(self, sim: typing.SupportsInt | typing.SupportsIndex) -> str:
        """error(self: mujoco._batch.Batch, sim: typing.SupportsInt | typing.SupportsIndex) -> str"""
    def expand(self, name: str) -> numpy.ndarray | None:
        """expand(self: object, name: str) -> numpy.ndarray | None"""
    def expanded(self, name: str) -> numpy.ndarray | None:
        """expanded(self: object, name: str) -> numpy.ndarray | None"""
    def forward(self, ids: numpy.typing.NDArray[numpy.int32] | None = ..., stage: typing.SupportsInt | typing.SupportsIndex = ...) -> int:
        """forward(self: mujoco._batch.Batch, ids: numpy.typing.NDArray[numpy.int32] | None = None, stage: typing.SupportsInt | typing.SupportsIndex = 3) -> int"""
    def get_data(self, sim: typing.SupportsInt | typing.SupportsIndex, data: mujoco._structs.MjData) -> None:
        """get_data(self: mujoco._batch.Batch, sim: typing.SupportsInt | typing.SupportsIndex, data: mujoco._structs.MjData) -> None"""
    def get_model(self, sim: typing.SupportsInt | typing.SupportsIndex, model: mujoco._structs.MjModel) -> None:
        """get_model(self: mujoco._batch.Batch, sim: typing.SupportsInt | typing.SupportsIndex, model: mujoco._structs.MjModel) -> None"""
    def jac(self, ids: numpy.typing.NDArray[numpy.int32] | None, body: numpy.typing.NDArray[numpy.int32], point: numpy.typing.NDArray[numpy.float64], jacp: numpy.typing.NDArray[numpy.float64] | None, jacr: numpy.typing.NDArray[numpy.float64] | None) -> int:
        """jac(self: mujoco._batch.Batch, ids: numpy.typing.NDArray[numpy.int32] | None, body: numpy.typing.NDArray[numpy.int32], point: numpy.typing.NDArray[numpy.float64], jacp: numpy.typing.NDArray[numpy.float64] | None, jacr: numpy.typing.NDArray[numpy.float64] | None) -> int"""
    def output(self, name: str) -> numpy.ndarray | None:
        """output(self: object, name: str) -> numpy.ndarray | None"""
    def ray(self, ids: numpy.typing.NDArray[numpy.int32] | None, pnt: numpy.typing.NDArray[numpy.float64], vec: numpy.typing.NDArray[numpy.float64], geomgroup: numpy.typing.NDArray[numpy.uint8] | None, flg_static: bool, bodyexclude: numpy.typing.NDArray[numpy.int32], dist: numpy.typing.NDArray[numpy.float64], geomid: numpy.typing.NDArray[numpy.int32], normal: numpy.typing.NDArray[numpy.float64] | None) -> int:
        """ray(self: mujoco._batch.Batch, ids: numpy.typing.NDArray[numpy.int32] | None, pnt: numpy.typing.NDArray[numpy.float64], vec: numpy.typing.NDArray[numpy.float64], geomgroup: numpy.typing.NDArray[numpy.uint8] | None, flg_static: bool, bodyexclude: numpy.typing.NDArray[numpy.int32], dist: numpy.typing.NDArray[numpy.float64], geomid: numpy.typing.NDArray[numpy.int32], normal: numpy.typing.NDArray[numpy.float64] | None) -> int"""
    def reset(self, ids: numpy.typing.NDArray[numpy.int32] | None = ..., keyframe: numpy.typing.NDArray[numpy.int32] | None = ..., stage: typing.SupportsInt | typing.SupportsIndex = ...) -> int:
        """reset(self: mujoco._batch.Batch, ids: numpy.typing.NDArray[numpy.int32] | None = None, keyframe: numpy.typing.NDArray[numpy.int32] | None = None, stage: typing.SupportsInt | typing.SupportsIndex = 3) -> int"""
    def rollout(self, ids: numpy.typing.NDArray[numpy.int32] | None, nstep: typing.SupportsInt | typing.SupportsIndex, control: numpy.typing.NDArray[numpy.float64] | None, control_spec: typing.SupportsInt | typing.SupportsIndex, records: collections.abc.Sequence[tuple]) -> int:
        """rollout(self: mujoco._batch.Batch, ids: numpy.typing.NDArray[numpy.int32] | None, nstep: typing.SupportsInt | typing.SupportsIndex, control: numpy.typing.NDArray[numpy.float64] | None, control_spec: typing.SupportsInt | typing.SupportsIndex, records: collections.abc.Sequence[tuple]) -> int"""
    def set_const(self, ids: numpy.typing.NDArray[numpy.int32] | None = ...) -> int:
        """set_const(self: mujoco._batch.Batch, ids: numpy.typing.NDArray[numpy.int32] | None = None) -> int"""
    def set_models(self, models: object) -> None:
        """set_models(self: mujoco._batch.Batch, models: object) -> None"""
    def state(self) -> numpy.ndarray:
        """state(self: object) -> numpy.ndarray"""
    def state_component(self, component: typing.SupportsInt | typing.SupportsIndex) -> tuple[int, int]:
        """state_component(self: mujoco._batch.Batch, component: typing.SupportsInt | typing.SupportsIndex) -> tuple[int, int]"""
    def status(self) -> numpy.ndarray:
        """status(self: object) -> numpy.ndarray"""
    def step(self, ids: numpy.typing.NDArray[numpy.int32] | None = ..., nstep: typing.SupportsInt | typing.SupportsIndex = ..., stage: typing.SupportsInt | typing.SupportsIndex = ...) -> int:
        """step(self: mujoco._batch.Batch, ids: numpy.typing.NDArray[numpy.int32] | None = None, nstep: typing.SupportsInt | typing.SupportsIndex = 1, stage: typing.SupportsInt | typing.SupportsIndex = 0) -> int"""
    def warning(self) -> numpy.ndarray:
        """warning(self: object) -> numpy.ndarray"""
    @property
    def nsim(self) -> int:
        """(arg0: mujoco._batch.Batch) -> int"""
    @property
    def nstate(self) -> int:
        """(arg0: mujoco._batch.Batch) -> int"""
    @property
    def nthread(self) -> int:
        """(arg0: mujoco._batch.Batch) -> int"""
    @property
    def persistent(self) -> bool:
        """(arg0: mujoco._batch.Batch) -> bool"""
