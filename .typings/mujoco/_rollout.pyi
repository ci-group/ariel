import numpy
import numpy.typing
import typing

class Rollout:
    def __init__(self, nthread: typing.SupportsInt) -> None:
        """__init__(self: mujoco._rollout.Rollout, *, nthread: typing.SupportsInt) -> None


        Construct a rollout object containing a thread pool for parallel rollouts.

          input arguments (optional):
            nthread            integer, number of threads in pool
                               if zero, this pool is not started and rollouts run on the calling thread

        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def rollout(self, model: list, data: list, nstep: typing.SupportsInt, control_spec: typing.SupportsInt, state0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], warmstart0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = ..., control: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = ..., state: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = ..., sensordata: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = ..., chunk_size: typing.SupportsInt | None = ...) -> None:
        """rollout(self: mujoco._rollout.Rollout, model: list, data: list, nstep: typing.SupportsInt, control_spec: typing.SupportsInt, state0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], warmstart0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = None, control: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = None, state: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = None, sensordata: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = None, chunk_size: typing.SupportsInt | None = None) -> None


        Roll out batch of trajectories from initial states, get resulting states and sensor values.

          input arguments (required):
            model              list of homogenous MjModel instances of length nbatch
            data               list of compatible MjData instances of length nthread
            nstep              integer, number of steps to be taken for each trajectory
            control_spec       specification of controls, ncontrol = mj_stateSize(m, control_spec)
            state0             (nbatch x nstate) nbatch initial state arrays, where
                                   nstate = mj_stateSize(m, mjSTATE_FULLPHYSICS)
          input arguments (optional):
            warmstart0         (nbatch x nv)                  nbatch qacc_warmstart arrays
            control            (nbatch x nstep x ncontrol)    nbatch trajectories of nstep controls
          output arguments (optional):
            state              (nbatch x nstep x nstate)      nbatch nstep states
            sensordata         (nbatch x nstep x nsendordata) nbatch trajectories of nstep sensordata arrays
            chunk_size         integer, determines threadpool chunk size. If unspecified, the default is
                                   chunk_size = max(1, nbatch / (nthread * 10))

        """
