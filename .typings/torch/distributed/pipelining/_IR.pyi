import torch
import torch.fx as fx
from ._backward import _null_coalesce_accumulate as _null_coalesce_accumulate, stage_backward as stage_backward
from ._unflatten import _outline_submodules as _outline_submodules
from ._utils import PipeInfo as PipeInfo
from .stage import _PipelineStage as _PipelineStage
from _typeshed import Incomplete
from enum import Enum
from torch.distributed import ProcessGroup as ProcessGroup
from torch.export import ExportedProgram as ExportedProgram
from torch.export.unflatten import InterpreterModule as InterpreterModule, _AttrKind as _AttrKind, _assign_attr as _assign_attr, _sink_params as _sink_params
from torch.fx.node import map_aggregate as map_aggregate
from torch.fx.passes.split_module import split_module as split_module
from typing import Any, Callable

logger: Incomplete

def _find_loss_from_output_and_spec(output_val, spec_val): ...
def _find_loss_output(mod: torch.nn.Module, g: fx.Graph, output_loss_value_spec): ...
def _insert_stage_symbolic_backward(g: fx.Graph, loss_node: fx.Node, output_node: fx.Node): ...

class PipeSequential(torch.nn.Sequential):
    @staticmethod
    def from_sequential(sequential_instance: torch.nn.Sequential): ...
    def forward(self, input): ...

class LossWrapper(torch.nn.Module):
    """
    LossWrapper is a convenient abstract class that allows you to wrap up both
    your model as well as its loss function and specify the connectivity between
    the inputs, model, loss function, and output value. Example::

        class MyModelWrapper(LossWrapper):
            def forward(self, x, targets):
                model_out = self.module(x)
                loss_value = self.loss_fn(model_out, targets)
                return loss_value

    The above example defines a connectivity where we expect the forward/loss/backward
    training procedure to take two arguments (x and targets), pass x into the module
    to get the output of the feedforward computation, pass the model output and the
    targets value into the loss function, and get and return the loss value, which will
    be backpropagated by PiPPy. The above class would then be instantiated like::

        model = ...  # instantiate the model
        loss_fn = torch.nn.MSELoss()  # for the sake of demonstration

        wrapper = MyModelWrapper(model, loss_fn)
        pipe = Pipe.from_tracing(wrapper, ...)

    """
    module: Incomplete
    loss_fn: Incomplete
    def __init__(self, module, loss_fn) -> None: ...
    def forward(self, *args, **kwargs) -> None: ...

class TrivialLossWrapper(LossWrapper):
    def forward(self, x, targets): ...
    loss_spec: bool

aten_pipe_split_alias: Incomplete

def pipe_split():
    """
    pipe_split is a special operator that is used to mark the boundary between
    stages in a module. It is used to split the module into stages. It is a
    no-op if your annotated module is run eagerly.

    Example:
        >>> # xdoctest: +SKIP
        >>> def forward(self, x):
        >>>     x = torch.mm(x, self.mm_param)
        >>>     x = torch.relu(x)
        >>>     pipe_split()
        >>>     x = self.lin(x)
        >>>     return x

    The above example will be split into two stages.
    """

class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2
MultiUseParamSpec = MultiUseParameterConfig | dict[str, MultiUseParameterConfig]

class DetachExecutor(fx.Interpreter):
    """
    Special interpreter to run the split_gm in testing that detaches all inputs to
    a module invocation. This is needed so that the values at the boundary are
    leaf modules in autograd execution.
    """
    value_remap: Incomplete
    def __init__(self, module, garbage_collect_values: bool = True) -> None: ...
    def run(self, *args, initial_env=None): ...
    def call_module(self, target, args, kwargs): ...
    def call_function(self, target, args, kwargs): ...

class _NodeReference:
    name: Incomplete
    def __init__(self, name) -> None: ...

class _LinearNodeList:
    serialize_node_list: Incomplete
    def __init__(self, node_list) -> None: ...
    def to_graph(self): ...

def _direct_serialization_deserialize(body, nodes):
    """
    Custom `__reduce__` method for serialization.
    DO AS I SAY -- NOT AS I DO. This violates the principle that
    GraphModules serialize via code export & re-tracing. We allow
    for this here because **PIPE STAGES SHOULD NOT BE PERSISTED
    TO DISK -- THIS IS ONLY FOR TRANSMISSION VIA RPC**. Persisting
    these instances to disk will expose internal implementation
    details of `fx.Graph` and related data structures and is
    NOT advised.
    """
def _direct_serialization_reduce(self): ...
def _modify_graph_op_device(gm: torch.fx.GraphModule, new_device: torch.device):
    '''
    Modify the device argument of all "call_function" nodes in the graph.  This
    is useful for moving the graph to a different device. In particular for
    generator ops, like torch.ones.
    '''

class Pipe(torch.nn.Module):
    split_gm: fx.GraphModule
    executor: DetachExecutor
    num_stages: int
    has_loss_and_backward: Incomplete
    loss_spec: Incomplete
    replicated_params: list[dict[str, str]]
    def __init__(self, split_gm: fx.GraphModule, num_stages: int, has_loss_and_backward: bool, loss_spec) -> None: ...
    def forward(self, *args, **kwargs): ...
    def get_stage_module(self, stage_idx: int) -> torch.nn.Module:
        """
        Return a stage module corresponding to `stage_idx` of the `pipe`.
        """
    @staticmethod
    def _number_and_count_forward_stages(gm: fx.GraphModule): ...
    @staticmethod
    def _from_traced(mod: torch.nn.Module, exported_program: ExportedProgram, multi_use_param_spec: MultiUseParamSpec | None = None, output_loss_value_spec=None, split_policy: Callable[[torch.fx.GraphModule], torch.fx.GraphModule] | None = None):
        """
        Additionally, the ``output_loss_value_spec`` value can be specified to disambiguate
        which value in the output of `forward` is the loss value on which PiPPy should apply
        backpropagation. For example, if your ``forward`` returns a tuple ``(loss, model_out)``,
        you can specify ``output_loss_value_spec=(True, False)``. Or, if your ``forward`` returns
        a dict ``{'loss': loss_value, 'model_out': model_out}``, you can specify
        ``output_loss_value_spec={'loss': True, 'model_out': False}``
        """
    def print_readable(self) -> None:
        """
        Print the pipe in a human-readable format.
        This will print both the root pipe and each stage module.
        """
    @staticmethod
    def _trace_with_export(mod: torch.nn.Module, example_args: tuple[Any, ...], example_kwargs: dict[str, Any] | None = None) -> ExportedProgram: ...
    @staticmethod
    def from_tracing(mod: torch.nn.Module, example_args: tuple[Any, ...], example_kwargs: dict[str, Any] | None = None, split_policy: Callable[[fx.GraphModule], fx.GraphModule] | None = None): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def info(self) -> PipeInfo:
        """
        Get information about the pipe.

        Returns
        -------
        PipeInfo
            A dataclass containing information about the pipe.
        """
    def build_stage(self, stage_index: int, device: torch.device, group: ProcessGroup | None = None) -> _PipelineStage:
        """
        Create a `PipelineStage` given a stage index and distributed group.
        The `PipelineStage` can run with `PipelineSchedule`s.
        """

class SplitPoint(Enum):
    """
    Enum representing the points at which a split can occur in the execution of a submodule.
    Attributes:
        BEGINNING: Represents adding a split point *before* the execution of a certain submodule in the `forward` function.
        END: Represents adding a split point *after* the execution of a certain submodule in the `forward` function.
    """
    BEGINNING = 1
    END = 2

class PipeSplitWrapper:
    SplitPoint = SplitPoint

def _split_before_forward(self, *args, **kwargs): ...
def _split_after_forward(self, *args, **kwargs): ...
def annotate_split_points(mod: torch.nn.Module, spec: dict[str, SplitPoint]): ...
def pipeline(module: torch.nn.Module, mb_args: tuple[Any, ...], mb_kwargs: dict[str, Any] | None = None, split_spec: dict[str, SplitPoint] | None = None, split_policy: Callable[[fx.GraphModule], fx.GraphModule] | None = None) -> Pipe:
    """
    Split a module based on a specification.

    See `Pipe` for more details.

    Arguments
    ---------
    module:
        The module to be split.
    mb_args:
        Example positional inputs, in micro-batch form.
    mb_kwargs:
        Example keyword inputs, in micro-batch form. (default: `None`)
    split_spec:
        A dictionary using submodule names as split marker. (default: `None`)
    split_policy:
        The policy to use for splitting the module. (default: `None`)

    Returns
    -------
    A pipeline representation of class `Pipe`.
    """
