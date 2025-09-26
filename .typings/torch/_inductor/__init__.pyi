import torch.fx
from .standalone_compile import CompiledArtifact
from torch._inductor.utils import InputType
from typing import Any, Literal

__all__ = ['compile', 'list_mode_options', 'list_options', 'cudagraph_mark_step_begin', 'standalone_compile']

def compile(gm: torch.fx.GraphModule, example_inputs: list[InputType], options: dict[str, Any] | None = None):
    """
    Compile a given FX graph with TorchInductor.  This allows compiling
    FX graphs captured without using TorchDynamo.

    Args:
        gm: The FX graph to compile.
        example_inputs:  List of tensor inputs.
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Callable with same behavior as gm but faster.
    """
def list_mode_options(mode: str | None = None, dynamic: bool | None = None) -> dict[str, Any]:
    """Returns a dictionary describing the optimizations that each of the available
    modes passed to `torch.compile()` performs.

    Args:
        mode (str, optional): The mode to return the optimizations for.
        If None, returns optimizations for all modes
        dynamic (bool, optional): Whether dynamic shape is enabled.

    Example::
        >>> torch._inductor.list_mode_options()
    """
def list_options() -> list[str]:
    """Returns a dictionary describing the optimizations and debug configurations
    that are available to `torch.compile()`.

    The options are documented in `torch._inductor.config`.

    Example::

        >>> torch._inductor.list_options()
    """
def cudagraph_mark_step_begin() -> None:
    """Indicates that a new iteration of inference or training is about to begin."""
def standalone_compile(gm: torch.fx.GraphModule, example_inputs: list[InputType], *, dynamic_shapes: Literal['from_example_inputs', 'from_tracing_context', 'from_graph'] = 'from_graph', options: dict[str, Any] | None = None) -> CompiledArtifact:
    '''
    Precompilation API for inductor.

    .. code-block:: python

        compiled_artifact = torch._inductor.standalone_compile(gm, args)
        compiled_artifact.save(path=path, format="binary")

        # Later on a new process
        loaded = torch._inductor.CompiledArtifact.load(path=path, format="binary")
        compiled_out = loaded(*args)

    Args:
        gm: Graph Module
        example_inputs: Inputs for the graph module
        dynamic_shapes: If "from_graph" (default), we will use the dynamic
            shapes in the passed-in graph module.
            If "from_tracing_context", we use the dynamic shape info in the
            ambient tracing context.
            If "from_example_inputs", we will specialize the graph on the
            example_inputs.
        options: Inductor compilation options

    Returns:
        CompiledArtifact that can be saved to disk or invoked directly.
    '''
