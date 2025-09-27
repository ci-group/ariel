import torch.fx as fx
import torch.nn as nn
from .aot_autograd import aot_function as aot_function, aot_module as aot_module, make_boxed_compiler as make_boxed_compiler
from .compile_utils import strip_overloads as strip_overloads
from .partitioners import default_partition as default_partition, draw_graph as draw_graph, min_cut_rematerialization_partition as min_cut_rematerialization_partition
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from torch import SymInt as SymInt
from torch._decomp import get_decompositions as get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols as bind_symbols
from typing import Callable

log: Incomplete

def _canonicalize(fx_g): ...
@contextmanager
def _disable_jit_autocast() -> Generator[None]: ...
@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps) -> Callable:
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """
def _draw_graph_compile(fx_g, _, name, clear_meta: bool = True): ...
def draw_graph_compile(name): ...
@make_boxed_compiler
def nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    Returns the :attr:`fx_g` Fx graph module as it is. This is a no-op compiler
    and can be used to check accuracy.

    .. warning::
        This API is experimental and likely to change.

    """

class DebugInterpreter(fx.Interpreter):
    symbol_mapping: Incomplete
    def run(self, *args) -> None: ...
    def run_node(self, n): ...

@make_boxed_compiler
def debug_nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    Returns a (slow) interpreter over the FX graph module that also checks
    various debugging properties (e.g., that tracing strides matched real
    strides.)
    """
@make_boxed_compiler
def simple_ts_compile(fx_g, _): ...
def nnc_jit(f): ...

aten: Incomplete
default_decompositions: Incomplete

@make_boxed_compiler
def print_compile(fx_g, _): ...
def memory_efficient_fusion(fn: Callable | nn.Module, **kwargs):
    """
    Wrapper function over :func:`aot_function` and :func:`aot_module` to perform
    memory efficient fusion. It uses the
    :func:`min_cut_rematerialization_partition` partitioner to perform efficient
    recomputation. It uses NVFuser to compile the generated forward and backward
    graphs.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Union[Callable, nn.Module]): A Python function or a ``nn.Module``
            that takes one ore more arguments. Must return one or more Tensors.
        **kwargs: Any other overrides you want to make to the settings

    Returns:
        Returns a ``Callable``  or ``nn.Module`` that retains the eager behavior
        of the original :attr:`fn`, but whose forward and backward graphs have
        gone through recomputation optimizations, and the graphs have been
        compiled with nvfuser.

    """
def debug_compile(fx_g, inps): ...

graph_index: int

def get_inputs(input_data_path):
    """
    Return a random input for the given inputs meta generated from _save_fx_default.
    """
def _save_fx_default(current_name, folder_name, dump_example_input, gm, example_inputs):
    """
    The forward, backward, and joint computation graph will be stored in
    {folder_name}/{current_name}/{current_name}_forward_{graph_index},
    {folder_name}/{current_name}/{current_name}_backward_{graph_index}, and
    {folder_name}/{current_name}/{current_name}_joint_{graph_index} respectively.
    The input shape of the graphs will be stored in the .input files.
    These files can be loaded with pickle,
    and is a list of format (type, shape, stride, dtype, device).
    In the case of type = int or float, it is just (type,).
    For joint graph input, it is a nested list [[],[]]
    where the two inner lists have the same format.
    If dump_example_input is True, example_inputs will be stored in .pt file.
    Since each function might produce multiple graphs,
    the graph_index is used to distinguish difference graphs
    """
def graph_dumper_aot(current_name, folder_name, dump_example_input: bool = False):
    """
    Dump the forward, backward, and joint computation graph.
    Example Usage:
    save_fx_func = graph_dumper_aot(current_name, folder_name, dump_example_input = False)
    optimize_ctx = torchdynamo.optimize(
        save_fx_func
    )
    with torch.enable_grad():
        with optimize_ctx:
            result = forward_and_backward_pass(model, example_inputs)
    """
