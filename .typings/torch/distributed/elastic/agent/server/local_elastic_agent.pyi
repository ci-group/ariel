import signal
import torch.distributed.elastic.timer as timer
from _typeshed import Incomplete
from torch.distributed.elastic.agent.server.api import RunResult, SimpleElasticAgent, WorkerGroup, WorkerSpec
from torch.distributed.elastic.agent.server.health_check_server import HealthCheckServer
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.multiprocessing import LogsSpecs, PContext
from typing import Any

__all__ = ['LocalElasticAgent', 'TORCHELASTIC_ENABLE_FILE_TIMER', 'TORCHELASTIC_TIMER_FILE', 'TORCHELASTIC_HEALTH_CHECK_PORT']

TORCHELASTIC_ENABLE_FILE_TIMER: str
TORCHELASTIC_HEALTH_CHECK_PORT: str
TORCHELASTIC_TIMER_FILE: str

class LocalElasticAgent(SimpleElasticAgent):
    '''An implementation of :py:class:`torchelastic.agent.server.ElasticAgent` that handles host-local workers.

    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.


    The worker function and argument passed to the worker function must be
    python multiprocessing compatible. To pass multiprocessing data structures
    to the workers you may create the data structure in the same multiprocessing
    context as the specified ``start_method`` and pass it as a function argument.

    The ``exit_barrier_timeout`` specifies the amount of time (in seconds) to wait
    for other agents to finish. This acts as a safety net to handle cases where
    workers finish at different times, to prevent agents from viewing workers
    that finished early as a scale-down event. It is strongly advised that the
    user code deal with ensuring that workers are terminated in a synchronous
    manner rather than relying on the exit_barrier_timeout.

    A named pipe based watchdog can be enabled in ```LocalElasticAgent``` if an
    environment variable ``TORCHELASTIC_ENABLE_FILE_TIMER`` with value 1 has
    been defined in the ```LocalElasticAgent``` process.
    Optionally, another environment variable ```TORCHELASTIC_TIMER_FILE```
    can be set with a unique file name for the named pipe. If the environment
    variable ```TORCHELASTIC_TIMER_FILE``` is not set, ```LocalElasticAgent```
    will internally create a unique file name and set it to the environment
    variable ```TORCHELASTIC_TIMER_FILE```, and this environment variable will
    be propagated to the worker processes to allow them to connect to the same
    named pipe that ```LocalElasticAgent``` uses.

    Logs are written to the specified log directory. Each log line will be by default
    prefixed by ``[${role_name}${local_rank}]:`` (e.g. ``[trainer0]: foobar``).
    Log prefixes can be customized by passing a `template string
    <https://docs.python.org/3/library/string.html#template-strings>`_ as the
    ``log_line_prefix_template`` argument.
    The following macros (identifiers) are substituted at runtime:
    ``${role_name}, ${local_rank}, ${rank}``. For example, to prefix each log line with
    global rank instead of the local rank, set ``log_line_prefix_template = "[${rank}]:``.


    Example launching function

    ::

        def trainer(args) -> str:
            return "do train"

        def main():
            start_method="spawn"
            shared_queue= multiprocessing.get_context(start_method).Queue()
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint=trainer,
                        args=("foobar",),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec, start_method)
            results = agent.run()

            if results.is_failed():
                print("trainer failed")
            else:
                print(f"rank 0 return value: {results.return_values[0]}")
                # prints -> rank 0 return value: do train

    Example launching binary

    ::

        def main():
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint="/usr/local/bin/trainer",
                        args=("--trainer-args", "foobar"),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec)
            results = agent.run()

            if not results.is_failed():
                print("binary launches do not have return values")

    '''
    _start_method: Incomplete
    _pcontext: PContext | None
    _rdzv_handler: Incomplete
    _log_line_prefix_template: Incomplete
    _worker_watchdog: timer.FileTimerServer | None
    _logs_specs: Incomplete
    _health_check_server: HealthCheckServer | None
    def __init__(self, spec: WorkerSpec, logs_specs: LogsSpecs, start_method: str = 'spawn', exit_barrier_timeout: float = 300, log_line_prefix_template: str | None = None) -> None: ...
    def _setup_local_watchdog(self, envs: dict[int, dict[str, str]]) -> None: ...
    @staticmethod
    def _get_current_time_secs() -> int: ...
    def _setup_healthcheck(self) -> None: ...
    def _get_fq_hostname(self) -> str: ...
    def _log_watchdog_event(self, name: str, request: timer.FileTimerRequest | None) -> None: ...
    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None: ...
    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]: ...
    def _shutdown(self, death_sig: signal.Signals = ...) -> None: ...
    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult: ...
