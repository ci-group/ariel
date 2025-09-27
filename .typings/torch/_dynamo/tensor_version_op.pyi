from _typeshed import Incomplete
from torch._prims import RETURN_TYPE as RETURN_TYPE, _make_prim as _make_prim
from torch._subclasses import FakeTensorMode as FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode as FunctionalTensorMode

_tensor_version: Incomplete

def _tensor_version_fake(fake_mode, self_tensor):
    """
    The initial dynamo capture of _tensor_version + _unsafe_set_version_counter turns the
    `._version` into an unbacked SymInt so that we don't need to specialize on the `._version`
    of input tensors to the graph.
    """

_unsafe_set_version_counter: Incomplete

def _tensor_version_functional(mode, self): ...
def _unsafe_set_version_counter_functional(ctx, tensors, versions) -> None: ...
