import dataclasses
from _typeshed import Incomplete
from torch import fx as fx
from torch._lazy import computation as computation
from torch._lazy.tensor_factory_functions import tensor_factory_functions as tensor_factory_functions
from typing import Any, Callable

debug: Incomplete

@dataclasses.dataclass
class GraphInputMatcher:
    """
    The GraphInputMatcher class setup the graph inputs for future calls after lazy tracing.
    Specifically, those graph inputs corresponding to method parameters should be replaced with the
    arguments for the current call.

    tensor_id_to_arg_idx maps the tensor id to the parameter index.
    graph_input_tensor_ids, graph_input_ivalues list the tensor_id and ivalue for each of the
    TS/XLA graph inputs.
    """
    tensor_id_to_arg_idx: dict[int, int]
    graph_input_tensor_ids: list[int]
    graph_input_ivalues: list[Any]
    def __call__(self, args): ...

class ReturnValueHandler:
    """
    When ltc_sync_multi is called on multi tensors, the compiled graph
    will contain output only for unique tensors - if a tensor appears multiple
    times in the input to _ltc_sync_multi, only the first occurance matters.

    However from python level, we still expect multi tensors returned with duplciation
    even if the TS graph dedup the output. e.g. for method:

      def forward(self, a):
        return a, a

    the TS graph captured by LTC will return a single tensor, but Python method expects 2.

    This class dedup the lazy tensors first to get the index that will be used
    to duplicate the eager tensors later.
    """
    index: list[list[int]]
    total_count: Incomplete
    def __init__(self, lazy_out_list) -> None: ...
    def duplicate_eager_tensors(self, eager_tensor_list): ...

def force_lazy_device(model: fx.GraphModule):
    """
    Factory methods in a Fx graph may create tensors for a specific eager devices.
    If we take no actions, those eager tensors will be mixed with lazy tensors and
    cause crash. This method overwrite those eager device to lazy device.
    """
def get_fallback_ops(): ...
def extract_compiled_graph(model: fx.GraphModule, example_inputs) -> Callable:
    """
    Optimize an eager model with LTC and returns a wrapper to execute the
    compiled graph directly without retracing. It depends on other mechanisms
    like TorchDynamo guards to guarantee the returned wrapper is only called
    when it's safe.
    """
