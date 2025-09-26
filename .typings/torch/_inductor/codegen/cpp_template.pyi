from .. import config as config, ir as ir
from ..autotune_process import CppBenchmarkRequest as CppBenchmarkRequest, TensorMeta as TensorMeta
from ..utils import IndentedBuffer as IndentedBuffer, Placeholder as Placeholder, unique as unique
from ..virtualized import V as V
from .common import KernelTemplate as KernelTemplate
from .cpp_template_kernel import CppTemplateCaller as CppTemplateCaller, CppTemplateKernel as CppTemplateKernel
from _typeshed import Incomplete
from typing import Callable

log: Incomplete

class CppTemplate(KernelTemplate):
    index_counter: Incomplete
    input_nodes: Incomplete
    index: Incomplete
    output_node: ir.Buffer | list[ir.Buffer]
    layout: Incomplete
    num_threads: Incomplete
    epilogue_creator: Incomplete
    def __init__(self, name: str, input_nodes, layout: ir.Layout, num_threads: int, epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None) -> None: ...
    def generate(self, **kwargs): ...
    def header(self) -> IndentedBuffer: ...
    def render(self, **kwargs) -> str: ...
