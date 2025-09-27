from _typeshed import Incomplete
from dataclasses import dataclass, field
from torch.distributed.elastic.multiprocessing import LogsSpecs
from typing import Any, Callable

__all__ = ['LaunchConfig', 'elastic_launch', 'launch_agent']

@dataclass
class LaunchConfig:
    '''
    Creates a rendezvous config.

    Args:
        min_nodes: Minimum amount of nodes that the user function will
                        be launched on. Elastic agent ensures that the user
                        function start only when the min_nodes amount enters
                        the rendezvous.
        max_nodes: Maximum amount of nodes that the user function
                        will be launched on.
        nproc_per_node: On each node the elastic agent will launch
                            this amount of workers that will execute user
                            defined function.
        rdzv_backend: rdzv_backend to use in the rendezvous (zeus-adapter, etcd).
        rdzv_endpoint: The endpoint of the rdzv sync. storage.
        rdzv_configs: Key, value pair that specifies rendezvous specific configuration.
        rdzv_timeout: Legacy argument that specifies timeout for the rendezvous. It is going
            to be removed in future versions, see the note below. The default timeout is 900 seconds.
        run_id: The unique run id of the job (if not passed a unique one will be
                deduced from run environment - flow workflow id in flow - or auto generated).
        role: User defined role of the worker (defaults to "trainer").
        max_restarts: The maximum amount of restarts that elastic agent will conduct
                    on workers before failure.
        monitor_interval: The interval in seconds that is used by the elastic_agent
                        as a period of monitoring workers.
        start_method: The method is used by the elastic agent to start the
                    workers (spawn, fork, forkserver).
        metrics_cfg: configuration to initialize metrics.
        local_addr: address of the local node if any. If not set, a lookup on the local
                machine\'s FQDN will be performed.
        local_ranks_filter: ranks for which to show logs in console. If not set, show from all.
        event_log_handler: name of the event logging handler as registered in
          `elastic/events/handlers.py <https://docs.pytorch.org/docs/stable/elastic/events.html>`_.


    .. note::
        `rdzv_timeout` is a legacy argument that will be removed in future.
        Set the timeout via `rdzv_configs[\'timeout\']`

    '''
    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    logs_specs: LogsSpecs | None = ...
    run_id: str = ...
    role: str = ...
    rdzv_endpoint: str = ...
    rdzv_backend: str = ...
    rdzv_configs: dict[str, Any] = field(default_factory=dict)
    rdzv_timeout: int = ...
    max_restarts: int = ...
    monitor_interval: float = ...
    start_method: str = ...
    log_line_prefix_template: str | None = ...
    metrics_cfg: dict[str, str] = field(default_factory=dict)
    local_addr: str | None = ...
    event_log_handler: str = ...
    def __post_init__(self) -> None: ...

class elastic_launch:
    '''
    Launches an torchelastic agent on the container that invoked the entrypoint.

        1. Pass the ``entrypoint`` arguments as non ``kwargs`` (e.g. no named parameters)/
           ``entrypoint`` can be a function or a command.
        2. The return value is a map of each worker\'s output mapped
           by their respective global rank.

    Usage

    ::

    def worker_fn(foo):
        # ...

    def main():
        # entrypoint is a function.
        outputs = elastic_launch(LaunchConfig, worker_fn)(foo)
        # return rank 0\'s output
        return outputs[0]

        # entrypoint is a command and ``script.py`` is the python module.
        outputs = elastic_launch(LaunchConfig, "script.py")(args)
        outputs = elastic_launch(LaunchConfig, "python")("script.py")
    '''
    _config: Incomplete
    _entrypoint: Incomplete
    def __init__(self, config: LaunchConfig, entrypoint: Callable | str | None) -> None: ...
    def __call__(self, *args): ...

def launch_agent(config: LaunchConfig, entrypoint: Callable | str | None, args: list[Any]) -> dict[int, Any]: ...
