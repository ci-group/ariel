from _typeshed import Incomplete
from argparse import ArgumentParser
from torch.distributed.argparse_util import check_env as check_env, env as env
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs as DefaultLogsSpecs, LogsSpecs as LogsSpecs, Std as Std
from torch.distributed.elastic.multiprocessing.errors import record as record
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config as _parse_rendezvous_config
from torch.distributed.elastic.utils import macros as macros
from torch.distributed.elastic.utils.logging import get_logger as get_logger
from torch.distributed.launcher.api import LaunchConfig as LaunchConfig, elastic_launch as elastic_launch
from torch.utils.backend_registration import _get_custom_mod_func as _get_custom_mod_func
from typing import Callable

logger: Incomplete

def get_args_parser() -> ArgumentParser:
    """Parse the command line options."""
def parse_args(args): ...
def parse_min_max_nnodes(nnodes: str): ...
def determine_local_world_size(nproc_per_node: str): ...
def get_rdzv_endpoint(args): ...
def get_use_env(args) -> bool:
    """
    Retrieve ``use_env`` from the args.

    ``use_env`` is a legacy argument, if ``use_env`` is False, the
    ``--node-rank`` argument will be transferred to all worker processes.
    ``use_env`` is only used by the ``torch.distributed.launch`` and will
    be deprecated in future releases.
    """
def _get_logs_specs_class(logs_specs_name: str | None) -> type[LogsSpecs]:
    """
    Attempts to load `torchrun.logs_spec` entrypoint with key of `logs_specs_name` param.
    Provides plugin mechanism to provide custom implementation of LogsSpecs.

    Returns `DefaultLogsSpecs` when logs_spec_name is None.
    Raises ValueError when entrypoint for `logs_spec_name` can't be found in entrypoints.
    """
def config_from_args(args) -> tuple[LaunchConfig, Callable | str, list[str]]: ...
def run_script_path(training_script: str, *training_script_args: str):
    '''
    Run the provided `training_script` from within this interpreter.

    Usage: `script_as_function("/abs/path/to/script.py", "--arg1", "val1")`
    '''
def run(args) -> None: ...
@record
def main(args=None) -> None: ...
