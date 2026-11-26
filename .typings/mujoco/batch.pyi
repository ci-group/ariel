import mujoco
import numpy as np
from _typeshed import Incomplete
from numpy import typing as npt
from typing import Any, Sequence

Ids: Incomplete
RecordItem: Incomplete

class SimulationError(RuntimeError):
    """MuJoCo raised an error in some simulations of a call.

  Those simulations kept the state they had before the call, and the others ran
  to completion. ids lists them; Batch.error(i) has each message.
  """
    ids: Incomplete
    def __init__(self, ids: np.ndarray, message: str, ncall: int | None = None) -> None: ...

class Batch:
    """nsim simulations of a copy of model, on nthread threads (0: all CPUs).

  With persistent=True, the batch keeps one MjData per simulation.

  A Batch must not be used from several Python threads at once: its calls are
  serialized, but each reports its failures from status and error(), which the
  next call overwrites.
  """
    model: Incomplete
    state: Incomplete
    warning: Incomplete
    status: Incomplete
    def __init__(self, model: mujoco.MjModel, nsim: int, nthread: int = 0, persistent: bool = False) -> None: ...
    def __getattr__(self, name: str) -> np.ndarray: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    @property
    def nsim(self) -> int: ...
    @property
    def nthread(self) -> int: ...
    @property
    def nstate(self) -> int: ...
    @property
    def persistent(self) -> bool: ...
    def error(self, sim: int) -> str:
        """Message of the error simulation sim raised in its last call, or ''."""
    def bind(self, name: str) -> np.ndarray:
        """Return a live (nsim, ...) array over an MjData field.

    State fields are views of the rows, which hold mjtNum (float64 in the
    standard build), so eq_active binds as mjtNum rather than as MjData's uint8.
    Other fields are outputs, copied out after every call, and read-only.

    Args:
      name: the MjData field, one of DATA_FIELDS.

    Returns:
      The (nsim, ...) array, cached so that each field is registered once.
    """
    def actuator(self, key: int | str) -> _Named: ...
    def body(self, key: int | str) -> _Named: ...
    def camera(self, key: int | str) -> _Named: ...
    def equality(self, key: int | str) -> _Named: ...
    def geom(self, key: int | str) -> _Named: ...
    def joint(self, key: int | str) -> _Named: ...
    def keyframe(self, key: int | str) -> _Named: ...
    def light(self, key: int | str) -> _Named: ...
    def material(self, key: int | str) -> _Named: ...
    def numeric(self, key: int | str) -> _Named: ...
    def pair(self, key: int | str) -> _Named: ...
    def sensor(self, key: int | str) -> _Named: ...
    def site(self, key: int | str) -> _Named: ...
    def tendon(self, key: int | str) -> _Named: ...
    def expand(self, name: str) -> np.ndarray:
        """Give a model field per-simulation values; return them, (nsim, ...).

    Array fields are named as in MjModel, MjOption fields as 'opt.<name>'.
    bvh_aabb holds the bodies' bounding volumes, which precede the meshes'.
    After writing a field that mj_setConst reads, one of
    mujoco.MjModel._setconst_inputs or an MjOption field, call set_const() so
    that the constants derived from it follow.

    Args:
      name: the MjModel field, one of MODEL_FIELDS.

    Returns:
      The live (nsim, ...) array of per-simulation values.
    """
    def expanded(self, name: str) -> np.ndarray | None:
        """Return a model field's read-only per-simulation values, or None.

    Returns the per-simulation values of a field that expand() or set_const()
    expanded, without expanding it if it is still shared.

    Args:
      name: the MjModel field, one of MODEL_FIELDS.

    Returns:
      The read-only (nsim, ...) array, or None if the field is not expanded.
    """
    def step(self, ids: Ids = None, nstep: int = 1, stage: int = ...) -> None:
        """Advance each simulation by nstep calls to mj_step, then compute stages.

    As with mj_step, derived fields describe the state before the last substep.
    stage computes the forward stages up to it at the new state: mjSTAGE_VEL
    makes position- and velocity-dependent fields and sensors current, as
    mj_step1 does; mjSTAGE_ACC runs mj_forward, whose acceleration-dependent
    values use controls that will not be applied. A persistent batch continues
    its next step from those stages, so that mjSTAGE_VEL costs nothing extra.

    Args:
      ids: the simulations, or None for all.
      nstep: number of substeps.
      stage: mjtStage, the stages to compute at the new state.
    """
    def forward(self, ids: Ids = None, stage: int = ...) -> None:
        """Compute the forward stages up to stage, by default mj_forward."""
    def reset(self, ids: Ids = None, keyframe: npt.ArrayLike = -1, stage: int = ...) -> None:
        """Reset to the model's defaults or to keyframes, then compute stages.

    Writes to the reset simulations' state made before the call are discarded.
    By default the reset runs mj_forward; stage=mjSTAGE_NONE skips it, for a
    reset that is followed by writes to the state and a forward().

    Args:
      ids: the simulations, or None for all.
      keyframe: a keyframe for every simulation of the call, or one each, (n,);
        -1 for the model's defaults.
      stage: mjtStage, the stages to compute after the reset.
    """
    def get_model(self, sim: int, model: mujoco.MjModel | None = None) -> mujoco.MjModel:
        """Simulation sim's model: the batch's, with its per-simulation fields.

    With caller models set (set_models), a copy of simulation sim's model.

    Args:
      sim: the simulation.
      model: an MjModel with the sizes of simulation sim's model to write into,
        or None for a copy of that model. The batch shares that model's assets
        (meshes, textures and so on) and writes everything else.

    Returns:
      The model, for rendering or for any mj_ function.
    """
    def set_models(self, models: None | mujoco.MjModel | Sequence[mujoco.MjModel]) -> None:
        """Run simulation i on models[i] instead of the batch's model.

    Each model must give MjData the layout of the batch's model: the same sizes
    of everything MjData holds, as models compiled from one spec with different
    parameters, or with meshes of other shapes, do. The batch keeps them alive
    and reads them; set them again after editing one. Caller models replace
    per-simulation fields: expand() and set_const() raise while they are set,
    and set_models raises if fields are expanded. Named views read model
    attributes from each simulation's model.

    Args:
      models: an MjModel per simulation, one MjModel for all of them, or None
        for the batch's model.
    """
    @property
    def models(self) -> None | mujoco.MjModel | tuple[mujoco.MjModel, ...]:
        """The caller models set with set_models, or None for the batch's model.

    An MjModel if one was set for every simulation, else a tuple with
    simulation i's model at index i.
    """
    def get_data(self, sim: int, data: mujoco.MjData | None = None) -> mujoco.MjData:
        """Simulation sim's state, and in a persistent batch its whole MjData.

    Derived fields are those of the simulation's last call in a persistent
    batch; otherwise run mj_forward with get_model(sim) to compute them.

    Args:
      sim: the simulation.
      data: an MjData with the batch's sizes to write into, or None for a new
        one.

    Returns:
      The data.
    """
    def set_const(self, ids: Ids = None) -> None:
        """Run mj_setConst per simulation; derived fields become per-simulation."""
    def rollout(self, nstep: int, control: npt.ArrayLike | None = None, control_spec: int = ..., record: Sequence[RecordItem] = ('qpos',), ids: Ids = None, out: dict[RecordItem, np.ndarray] | None = None) -> dict[RecordItem, np.ndarray]:
        """Advance each simulation by nstep substeps, recording after each.

    As mujoco.rollout: before substep k, the state components in control_spec
    are set from control, (n, nstep, ncontrol) for the n simulations of the
    call, or the inputs are left as they are when control is None. A simulation
    whose warning counters rise stops stepping, and its remaining records repeat
    its last ones.

    Args:
      nstep: number of substeps.
      control: (n, nstep, mj_stateSize(model, control_spec)) or None.
      control_spec: mjtState bits within mjSTATE_USER.
      record: MjData field names, or mjtState specs for state vectors. Fields
        are recorded with MjData's types, so eq_active is uint8 here though
        bind() gives it as mjtNum.
      ids: the simulations, or None for all.
      out: arrays to record into, by record item, of the shapes and dtypes this
        call returns; the items not in it get new arrays.

    Returns:
      A dict from each record item to its (n, nstep, ...) array.
    """
    def jac(self, body: npt.ArrayLike, point: npt.ArrayLike, ids: Ids = None) -> tuple[np.ndarray, np.ndarray]:
        """Translational and rotational Jacobians of a world point on a body.

    Runs mj_kinematics and mj_comPos per simulation, from its state, then
    mj_jac; the state and bound fields are untouched. In a persistent batch the
    simulation's MjData keeps these kinematics, as get_data() shows, and its
    next step starts afresh rather than continuing.

    Args:
      body: body id, or one per simulation of the call, (n,).
      point: world-frame points, (3,) or (n, 3) for the n simulations of the
        call.
      ids: the simulations, or None for all.

    Returns:
      jacp, jacr: (n, 3, nv) each.
    """
    def ray(self, pnt: npt.ArrayLike, vec: npt.ArrayLike, geomgroup: npt.ArrayLike | None = None, flg_static: bool = True, bodyexclude: npt.ArrayLike = -1, ids: Ids = None, normal: bool = False) -> tuple[np.ndarray, ...]:
        """Intersect rays with each simulation's geoms, as mj_ray.

    Runs mj_kinematics (and mj_flex) per simulation, from its state; the state
    and bound fields are untouched. In a persistent batch the simulation's
    MjData keeps these kinematics, as get_data() shows, and its next step starts
    afresh rather than continuing.

    Args:
      pnt: ray origins, broadcast to (n, nray, 3) for the n simulations of the
        call: (3,) is one ray, (nray, 3) the same rays in every simulation.
      vec: ray directions, broadcast likewise.
      geomgroup: geom groups to include, (mjNGROUP,), or None for all.
      flg_static: whether to include static geoms.
      bodyexclude: body whose geoms a ray ignores, or -1; broadcast to (n,
        nray).
      ids: the simulations, or None for all.
      normal: whether to return the surface normal at each hit.

    Returns:
      dist, geomid: (n, nray) each; dist is -1 and geomid -1 for a miss. With
      normal, also the normals, (n, nray, 3), zero for a miss.
    """

class _Named:
    """One object's slices of per-simulation fields, as MjData's and MjModel's.

  batch.joint('hinge').qpos is the part of batch.bind('qpos') that
  data.joint('hinge').qpos is of data.qpos, and batch.body('pole').mass the part
  of batch.expand('body_mass') that model.body('pole').mass is of
  model.body_mass. Data attributes bind their field. Model attributes read
  without expanding: the per-simulation values of a field that expand() or
  set_const() expanded, or else the model's value for every simulation,
  read-only unless expanded by expand(). Assignment expands the field and writes
  every simulation's value. Structural fields (addresses, counts, ids and types)
  are the same in every simulation and cannot be assigned. Equalities, pairs,
  materials, numerics and keyframes have only MjModel attributes.
  """
    id: int
    name: str
    def __init__(self, batch: Batch, kind: str, key: int | str, data: bool = True) -> None: ...
    def __getattr__(self, attr: str) -> np.ndarray: ...
    def __setattr__(self, attr: str, value: Any) -> None:
        """Write value to every simulation's slice: batch.joint('j').qpos = 0.5."""

DATA_FIELDS: Incomplete
MODEL_FIELDS: Incomplete
