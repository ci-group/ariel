import contextlib
import dataclasses
import functools
import torch
from . import config as config, ir as ir
from .scheduler import BaseSchedulerNode as BaseSchedulerNode, FusedSchedulerNode as FusedSchedulerNode, NopKernelSchedulerNode as NopKernelSchedulerNode, OutputNode as OutputNode, SchedulerNode as SchedulerNode
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Iterator
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro as save_graph_repro
from torch._dynamo.utils import get_debug_dir as get_debug_dir
from torch._logging import getArtifactLogger as getArtifactLogger
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.passes.shape_prop import TensorMetadata as TensorMetadata, _extract_tensor_metadata as _extract_tensor_metadata
from torch.fx.passes.tools_common import legalize_graph as legalize_graph
from torch.types import FileLike as FileLike
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._pytree import tree_map as tree_map
from typing import Any, Callable, IO, NamedTuple

log: Incomplete
ir_pre_fusion_log: Incomplete
ir_post_fusion_log: Incomplete
SchedulerNodeList = list[Any]

class BufMeta(NamedTuple):
    name: Incomplete
    n_origin: Incomplete

GRAPHVIZ_COMMAND_SCALABLE: Incomplete

@functools.cache
def has_dot() -> bool: ...
def draw_buffers(nodes: list[BaseSchedulerNode], print_graph: bool = False, fname: str | None = None) -> None:
    """
    Draw a graph in fname.svg.
    """
def create_fx_from_snodes(snodes: list[BaseSchedulerNode]) -> fx.Graph:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """
def update_orig_fx_node_name_to_buf_name(nodes: SchedulerNodeList | None, node_name_to_buf_name: dict[str, str], parent_buf_name: str | None = None, n_origins: int = 0) -> None: ...
def get_node_name_to_buf_meta(node_name_to_buf_name: dict[str, str]) -> dict[str, BufMeta]: ...
def annotate_orig_fx_with_snodes(gm: torch.fx.GraphModule, snodes: SchedulerNodeList) -> None:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """
@contextlib.contextmanager
def enable_aot_logging() -> Iterator[None]: ...

_inductor_post_to_pre_grad_nodes: dict[str, Any]
_pre_grad_graph_id: int | None

class DebugContext:
    _counter: Incomplete
    _inductor_triton_kernel_to_post_grad_node_info: dict[str, list[str]]
    @staticmethod
    def create_debug_dir(folder_name: str) -> str | None: ...
    _prof: Incomplete
    _path: Incomplete
    _stack: Incomplete
    def __init__(self) -> None: ...
    def copy(self, new_path: str) -> None: ...
    def fopen(self, filename: str, write_mode: str = 'w', *args: Any, **kwargs: Any) -> IO[Any]: ...
    @contextlib.contextmanager
    def fopen_context(self, filename: str, write_mode: str = 'w', *args: Any, **kwargs: Any) -> Iterator[IO[Any]]: ...
    def filename(self, suffix: str) -> str: ...
    def upload_tar(self) -> None: ...
    def __enter__(self) -> None: ...
    def _setup_log_capture(self, filename: str, level: int) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None) -> None: ...
    def _save_profile_data(self) -> None: ...
    def __getattr__(self, name: str) -> Callable[..., None] | None: ...

class DebugFormatter:
    fopen: Incomplete
    fopen_context: Incomplete
    filename: Incomplete
    handler: Incomplete
    def __init__(self, handler: DebugContext) -> None: ...
    def fx_graph(self, gm: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> None: ...
    def fx_graph_transformed(self, gm: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> None: ...
    def ir_pre_fusion(self, nodes: SchedulerNodeList) -> None: ...
    def ir_post_fusion(self, nodes: SchedulerNodeList) -> None: ...
    @staticmethod
    def _write_ir(nodes: SchedulerNodeList) -> str: ...
    def graph_diagram(self, nodes: SchedulerNodeList) -> None: ...
    def draw_orig_fx_graph(self, gm: torch.fx.GraphModule, nodes: SchedulerNodeList) -> None: ...
    def output_code(self, filename: str, extension: str = 'py') -> None: ...
    def log_inductor_triton_kernel_to_post_grad_node_info(self, filename: str = 'inductor_generated_kernel_to_post_grad_nodes.json') -> tuple[dict[str, list[str]], dict[str, Any]]: ...
    def log_autotuning_results(self, name: str, input_nodes: list[ir.IRNode], timings: dict['ChoiceCaller', float], elapse: float, precompile_elapse: float, prescreening_elapse: float | None) -> None: ...

def log_ir_pre_fusion(nodes: SchedulerNodeList) -> None: ...
def log_ir_post_fusion(nodes: SchedulerNodeList) -> None: ...

@dataclasses.dataclass
class TensorMetadataHolder:
    tensor_metadata: TensorMetadata
    device: torch.device

save_args_cnt: Incomplete

def create_node_mapping(pre_grad_graph_id: int, post_to_pre_grad_nodes_json: dict[str, Any], triton_kernel_to_post_grad_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Create bidirectional mappings between:

    - pre_grad graph nodes and post_grad graph code nodes, and vice versa
    - triton kernel name and post_grad graph code nodes, and vice versa
    """
def save_args_for_compile_fx_inner(*args: Any, **kwargs: Any) -> None:
    """
    This function is used to save arguments for a compile_fx_inner function call
    to the file system.  Later on one can replay the compile_fx_inner call
    with the saved arguments using load_args_and_run_compile_fx_inner.
    """
def load_args_and_run_compile_fx_inner(path: str) -> Any: ...
def aot_inductor_minifier_wrapper(func: Callable[..., str], exported_program: torch.export.ExportedProgram, *, inductor_configs: dict[str, Any], package_path: FileLike | None = None) -> str: ...
