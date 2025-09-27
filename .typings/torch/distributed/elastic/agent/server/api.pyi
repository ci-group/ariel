import abc
import signal
import torch.distributed.elastic.rendezvous as rdzv
from _typeshed import Incomplete
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from torch.distributed.elastic.events import Event, EventSource
from torch.distributed.elastic.metrics import prof
from torch.distributed.elastic.multiprocessing import ProcessFailure
from typing import Any, Callable

__all__ = ['WorkerSpec', 'Worker', 'WorkerState', 'WorkerGroup', 'RunResult', 'ElasticAgent', 'SimpleElasticAgent']

@dataclass
class WorkerSpec:
    """Blueprint information about a particular type of worker.

    For a given role, there must only exist a single worker spec.
    Worker spec is expected to be homogeneous across all nodes (machine),
    that is each node runs the same number of workers for a particular spec.

    Args:
        role: user-defined role for the workers with this spec
        local_world_size: number local workers to run
        fn: (deprecated use entrypoint instead)
        entrypoint: worker function or command
        args: arguments to pass to ``entrypoint``
        rdzv_handler: handles rdzv for this set of workers
        max_restarts: number of max retries for the workers
        monitor_interval: monitor status of workers every ``n`` seconds
        master_port: fixed port to run the c10d store on rank 0
                     if not specified then will chose a random free port
        master_addr: fixed master_addr to run the c10d store on rank 0
                     if not specified then will chose hostname on agent rank 0
        redirects: redirect std streams to a file,
                   selectively redirect for a particular
                   local rank by passing a map
        tee: tees the specified std stream(s) to console + file,
             selectively tee for a particular local rank by passing a map,
             takes precedence over ``redirects`` settings.
        event_log_handler: name of the event logging handler as registered in
          `elastic/events/handlers.py <https://docs.pytorch.org/docs/stable/elastic/events.html>`_.
    """
    role: str
    local_world_size: int
    rdzv_handler: rdzv.RendezvousHandler
    fn: Callable | None = ...
    entrypoint: Callable | str | None = ...
    args: tuple = ...
    max_restarts: int = ...
    monitor_interval: float = ...
    master_port: int | None = ...
    master_addr: str | None = ...
    local_addr: str | None = ...
    event_log_handler: str = ...
    def __post_init__(self) -> None: ...
    def get_entrypoint_name(self):
        """Get the entry point name.

        If the entrypoint is a function (e.g. ``Callable``) returns its ``__qualname__``
        else if the entrypoint is a binary (e.g. ``str``), returns the binary name.
        """

class Worker:
    """A worker instance.

    Contrast this with ``WorkerSpec`` that represents the specifications of a
    worker. A ``Worker`` is created from a ``WorkerSpec``. A ``Worker`` is to
    a ``WorkerSpec`` as an object is to a class.

    The ``id`` of the worker is interpreted
    by the specific implementation of ``ElasticAgent``. For a local
    agent, it could be the ``pid (int)`` of the worker, for a remote
    agent it could be encoded as ``host:port (string)``.

    Args:
        id (Any): uniquely identifies a worker (interpreted by the agent)
        local_rank (int): local rank of the worker
        global_rank (int): global rank of the worker
        role_rank (int): rank of the worker across all workers that have the same role
        world_size (int): number of workers (globally)
        role_world_size (int): number of workers that have the same role
    """
    __slots__: Incomplete
    id: Any
    local_rank: int
    global_rank: int
    role_rank: int
    world_size: int
    role_world_size: int
    def __init__(self, local_rank: int, global_rank: int = -1, role_rank: int = -1, world_size: int = -1, role_world_size: int = -1) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class WorkerState(str, Enum):
    """A state of the ``WorkerGroup``.

    Workers in a worker group change state as a unit. If a single worker
    in a worker group fails the entire set is considered failed::

      UNKNOWN - agent lost track of worker group state, unrecoverable
      INIT - worker group object created not yet started
      HEALTHY - workers running and healthy
      UNHEALTHY - workers running and unhealthy
      STOPPED - workers stopped (interrupted) by the agent
      SUCCEEDED - workers finished running (exit 0)
      FAILED - workers failed to successfully finish (exit !0)


    A worker group starts from an initial ``INIT`` state,
    then progresses to ``HEALTHY`` or ``UNHEALTHY`` states,
    and finally reaches a terminal ``SUCCEEDED`` or ``FAILED`` state.

    Worker groups can be interrupted and temporarily put into ``STOPPED`` state
    by the agent. Workers in ``STOPPED`` state are scheduled to be restarted
    in the near future by the agent. Some examples of workers being put into
    ``STOPPED`` state are:

    1. Worker group failure|unhealthy observed
    2. Membership change detected

    When actions (start, stop, rdzv, retry, etc) on worker group fails
    and results in the action being partially applied to the worker group
    the state will be ``UNKNOWN``. Typically this happens on uncaught/unhandled
    exceptions during state change events on the agent. The agent is not
    expected to recover worker groups in ``UNKNOWN`` state and is better off
    self terminating and allowing the job manager to retry the node.
    """
    UNKNOWN = 'UNKNOWN'
    INIT = 'INIT'
    HEALTHY = 'HEALTHY'
    UNHEALTHY = 'UNHEALTHY'
    STOPPED = 'STOPPED'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'
    @staticmethod
    def is_running(state: WorkerState) -> bool:
        """Return the state of the Worker.

        Returns:
             True if the worker state represents workers still running
             (e.g. that the process exists but not necessarily healthy).
        """

class WorkerGroup:
    """A set of ``Worker`` instances.

    The class defines a set of ``Worker`` instances for the given ``WorkerSpec`` managed by ``ElasticAgent``. Whether the worker
    group contains cross instance workers or not depends on the implementation of the agent.
    """
    __slots__: Incomplete
    spec: Incomplete
    workers: Incomplete
    store: Incomplete
    group_rank: Incomplete
    group_world_size: Incomplete
    master_addr: Incomplete
    master_port: Incomplete
    state: Incomplete
    def __init__(self, spec: WorkerSpec) -> None: ...

class _RoleInstanceInfo:
    """The class is used by the agent to exchange the information with other agents.

    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """
    __slots__: Incomplete
    role: Incomplete
    rank: Incomplete
    local_world_size: Incomplete
    def __init__(self, role: str, rank: int, local_world_size: int) -> None:
        """Initialize the agent class instance.

        Args:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """
    def serialize(self) -> bytes: ...
    @staticmethod
    def deserialize(data: bytes): ...
    @staticmethod
    def compare(obj1, obj2) -> int: ...
    @staticmethod
    def find_role_boundaries(roles_infos: list, role: str) -> tuple[int, int]: ...

@dataclass
class RunResult:
    '''Return results of the worker executions.

    Run results follow an "all-or-nothing" policy where the run is successful if and
    only if ALL local workers managed by this agent complete successfully.

    If the result is successful (e.g. ``is_failed() = False``) then the ``return_values``
    field contains the outputs (return values) of the workers managed by THIS agent mapped
    by their GLOBAL ranks. That is ``result.return_values[0]`` is the return value of
    global rank 0.

    .. note:: ``return_values`` are only meaningful for when the worker entrypoint
              is a function. Workers specified as a binary entrypoint do not canonically
              have a return value and the ``return_values`` field is meaningless and
              may be empty.

    If ``is_failed()`` returns ``True`` then the ``failures`` field contains the
    failure information, again, mapped by the GLOBAL rank of the worker that failed.

    The keys in ``return_values`` and ``failures`` are mutually exclusive, that is,
    a worker\'s final state can only be one of: succeeded, failed. Workers intentionally
    terminated by the agent according to the agent\'s restart policy, are not represented
    in either ``return_values`` nor ``failures``.
    '''
    state: WorkerState
    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    def is_failed(self) -> bool: ...

class ElasticAgent(abc.ABC, metaclass=abc.ABCMeta):
    '''An agent process responsible for managing one or more worker processes.

    The worker processes are assumed to be regular distributed PyTorch scripts.
    When the worker process is created by the agent, the agent provides the
    necessary information for the worker processes to properly initialize
    a torch process group.

    The exact deployment topology and ratio of agent-to-worker is dependent
    on the specific implementation of the agent and the user\'s job placement
    preferences. For instance, to run a distributed training job on GPU with
    8 trainers (one per GPU) one can:

    1. Use 8 x single GPU instances, place an agent per instance, managing
       1 worker per agent.
    2. Use 4 x double GPU instances, place an agent per instance, managing
       2 workers per agent.
    3. Use 2 x quad GPU instances, place an agent per instance, managing
       4 workers per agent.
    4. Use 1 x 8 GPU instance, place an agent per instance, managing
       8 workers per agent.

    Usage
    ::

     group_result = agent.run()
      if group_result.is_failed():
        # workers failed
        failure = group_result.failures[0]
        logger.exception("worker 0 failed with exit code : %s", failure.exit_code)
      else:
        return group_result.return_values[0] # return rank 0\'s results

    '''
    @abc.abstractmethod
    def run(self, role: str = ...) -> RunResult:
        """Run the agent.

        Supports retrying the worker group on failures up to ``max_restarts``.

        Returns:
            The result of the execution, containing the return values or
            failure details for each worker mapped by the worker's global rank.

        Raises:
            Exception - any other failures NOT related to worker process
        """
    @abc.abstractmethod
    def get_worker_group(self, role: str = ...) -> WorkerGroup:
        """Return the ``WorkerGroup`` for the given ``role``.

        Note that the worker group is a mutable object and hence in a
        multi-threaded/process environment it may change state.
        Implementers are encouraged (but not required) to return
        a defensive read-only copy.
        """

class SimpleElasticAgent(ElasticAgent, metaclass=abc.ABCMeta):
    """An ``ElasticAgent`` that manages one particular type of worker role.

    An ``ElasticAgent`` that manages workers (``WorkerGroup``) for a single ``WorkerSpec``
    such as one particular type of worker role.
    """
    _worker_group: Incomplete
    _remaining_restarts: Incomplete
    _store: Incomplete
    _exit_barrier_timeout: Incomplete
    _total_execution_time: int
    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float = 300) -> None: ...
    def get_worker_group(self, role: str = ...) -> WorkerGroup: ...
    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        """Start ``worker_group.spec.local_world_size`` number of workers.

        This is according to worker spec for the worker group .
        Returns a map of ``local_rank`` to worker ``id``.
        """
    @abc.abstractmethod
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        """Stop all workers in the given worker group.

        Implementers must deal with workers in all states defined by
        ``WorkerState``. That is, it must gracefully handle stopping
        non-existent workers, unhealthy (stuck) workers, etc.
        """
    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        """Check on the workers for the ``worker_group``.

        This function also returns the new state of the worker group.
        """
    @abc.abstractmethod
    def _shutdown(self, death_sig: signal.Signals = ...) -> None:
        """Clean up any resources that were allocated during the agent's work.

        Args:
            death_sig: Signal to send to the child process, SIGTERM is default
        """
    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        """Run rendezvous for the workers specified by the worker spec.

        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """
    @prof
    def _assign_worker_ranks(self, store, group_rank: int, group_world_size: int, spec: WorkerSpec) -> list[Worker]:
        """Determine proper ranks for worker processes.

        Fast Path: when all workers have the same role and world size. We calculate
        the global rank to be group_rank * group_world_size + local_rank. And the
        `role_world_size` is the same as `global_world_size`. No TCP store is used in
        this case. This is only enabled when users set the environment variable
        `TORCH_ELASTIC_WORKER_IDENTICAL` to 1.

        Time complexity: each worker O(1), overall O(1)

        Slow Path: when workers have different roles and world sizes. We use the
        the following algorithm:

        1. Each agent writes its configuration(group_rank, group_world_size
           , num_workers) to the common store.
        2. The rank 0 agent reads all the role_info from the store and
           determines each agents worker ranks.
        3. Determine the global rank: the global rank of the workers is computed
           by cumulative sum of the local_world_size for all workers in front of it.
           For efficiency reasons each worker is assigned a base global rank
           such that it's workers are in the range [base_global_rank,
           base_global_rank + local_world_size).
        4. Determine the role rank: The role rank is determined using the algorithms
           in the point 3 with the exception that the ranks are calculated with
           respect to the role name.
        5. The rank 0 agent writes the assigned ranks to the store.
        6. Each agent reads the assigned ranks from the store.

        Time complexity: each worker O(1), rank0 O(n), overall O(n)
        """
    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        """Start a fresh set of workers for the worker_group.

        Essentially, a rendezvous followed by a ``start_workers``.
        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
    @prof
    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        """Restart (stops, rendezvous, starts) all local workers in the group."""
    @prof
    def run(self, role: str = ...) -> RunResult: ...
    def get_event_failed(self) -> Event: ...
    def get_event_succeeded(self) -> Event: ...
    def _record_worker_events(self, result: RunResult) -> None: ...
    def _get_worker_state(self, worker: Worker, result: RunResult) -> str: ...
    @contextmanager
    def record_duration(self, state: str): ...
    def _construct_event(self, state: str, source: EventSource, worker: Worker | None = None, raw_error: str | None = None, duration_ms: float | None = None) -> Event: ...
    def _record_metrics(self, group_results: RunResult): ...
    def _record_metric_with_condition(self, metric_name, condition) -> None: ...
    def _record_flakiness_metric(self, is_failed: bool = False): ...
    def _invoke_run(self, role: str = ...) -> RunResult: ...
    def _exit_barrier(self) -> None:
        """
        Define a barrier that keeps the agent process alive until all workers finish.

        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times.
        """
