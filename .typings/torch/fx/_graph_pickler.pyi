import abc
import dataclasses
import io
import pickle
import torch
from _typeshed import Incomplete
from abc import abstractmethod
from torch._guards import TracingContext as TracingContext
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor, FakeTensorMode as FakeTensorMode, Tensor as Tensor
from torch._subclasses.meta_utils import MetaConverter as MetaConverter, MetaTensorDesc as MetaTensorDesc, MetaTensorDescriber as MetaTensorDescriber
from torch.fx.experimental.sym_node import SymNode as SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv
from torch.utils._mode_utils import no_dispatch as no_dispatch
from typing import Any, Callable, TypeVar
from typing_extensions import Self, override

_SymNodeT = TypeVar('_SymNodeT', torch.SymInt, torch.SymFloat)

def _ops_filter_safe(name: str) -> bool:
    """
    An ops filter which allows pickle-safe ops. Pickle-safe ops are built-in
    ones where it will be possible to unpickle on any machine which has PyTorch.
    """

@dataclasses.dataclass
class Options:
    ops_filter: Callable[[str], bool] | None = ...

class GraphPickler(pickle.Pickler):
    """
    GraphPickler is a Pickler which helps pickling fx graph - in particular
    GraphModule.
    """
    options: Incomplete
    _unpickle_state: Incomplete
    _meta_tensor_describer: Incomplete
    def __init__(self, file: io.BytesIO, options: Options | None = None) -> None: ...
    @override
    def reducer_override(self, obj: object) -> tuple[Callable[..., Any], tuple[Any, ...]]: ...
    @override
    def persistent_id(self, obj: object) -> str | None: ...
    @classmethod
    def dumps(cls, obj: object, options: Options | None = None) -> bytes:
        """
        Pickle an object.
        """
    @staticmethod
    def loads(data: bytes, fake_mode: FakeTensorMode) -> object:
        """
        Unpickle an object.
        """

class _UnpickleState:
    fake_mode: Incomplete
    meta_converter: MetaConverter[FakeTensor]
    def __init__(self, fake_mode: FakeTensorMode) -> None: ...

_UnpickleStateToken: Incomplete

class _GraphUnpickler(pickle.Unpickler):
    _unpickle_state: Incomplete
    def __init__(self, stream: io.BytesIO, unpickle_state: _UnpickleState) -> None: ...
    @override
    def persistent_load(self, pid: object) -> object: ...

class _ShapeEnvPickleData:
    data: dict[str, object]
    @classmethod
    def reduce_helper(cls, pickler: GraphPickler, obj: ShapeEnv) -> tuple[Callable[[Self, _UnpickleState], ShapeEnv], tuple[Self, _UnpickleStateToken]]: ...
    def __init__(self, env: ShapeEnv) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> ShapeEnv: ...

class _SymNodePickleData:
    @classmethod
    def reduce_helper(cls, pickler: GraphPickler, obj: _SymNodeT) -> tuple[Callable[[Self, _UnpickleState], _SymNodeT], tuple[Self, _UnpickleStateToken]]: ...
    expr: Incomplete
    shape_env: Incomplete
    pytype: Incomplete
    hint: Incomplete
    def __init__(self, node: SymNode) -> None: ...
    def _to_sym_node(self) -> SymNode: ...
    def unpickle_sym_int(self, unpickle_state: _UnpickleState) -> torch.SymInt: ...

class _TensorPickleData:
    metadata: MetaTensorDesc[FakeTensor]
    @classmethod
    def reduce_helper(cls, pickler: GraphPickler, obj: FakeTensor) -> tuple[Callable[[Self, _UnpickleState], FakeTensor], tuple[Self, _UnpickleStateToken]]: ...
    def __init__(self, describer: MetaTensorDescriber, t: Tensor) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> FakeTensor: ...

class _TorchNumpyPickleData:
    @classmethod
    def reduce_helper(cls, pickler: GraphPickler, obj: object) -> tuple[Callable[[Self, _UnpickleState], object], tuple[Self, _UnpickleStateToken]] | None: ...
    mod: Incomplete
    name: Incomplete
    def __init__(self, mod: str, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> Callable[..., object]: ...
    @classmethod
    def from_object(cls, tnp: object) -> Self | None: ...

class _GraphModulePickleData:
    @classmethod
    def reduce_helper(cls, pickler: GraphPickler, obj: torch.fx.GraphModule) -> tuple[Callable[[Self, _UnpickleState], torch.fx.GraphModule], tuple[Self, _UnpickleStateToken]]: ...
    gm_dict: Incomplete
    graph: Incomplete
    def __init__(self, gm: torch.fx.GraphModule, options: Options) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> torch.fx.GraphModule: ...

class _NodePickleData:
    args: Incomplete
    kwargs: Incomplete
    name: Incomplete
    op: Incomplete
    target: Incomplete
    type: Incomplete
    meta: Incomplete
    def __init__(self, node: torch.fx.Node, mapping: dict[torch.fx.Node, '_NodePickleData'], options: Options) -> None: ...
    def unpickle(self, graph: torch.fx.Graph, mapping: dict['_NodePickleData', torch.fx.Node], unpickle_state: _UnpickleState) -> torch.fx.Node: ...

class _OpPickleData(metaclass=abc.ABCMeta):
    @classmethod
    def reduce_helper(cls, pickler: GraphPickler, op: object) -> tuple[Callable[[_UnpickleState], object], tuple[_UnpickleStateToken]]: ...
    @classmethod
    def pickle(cls, op: object, options: Options) -> _OpPickleData: ...
    @staticmethod
    def _pickle_op(name: str, datacls: type['_OpOverloadPickleData'] | type['_OpOverloadPacketPickleData'], options: Options) -> _OpPickleData: ...
    @abstractmethod
    def unpickle(self, unpickle_state: _UnpickleState) -> object: ...
    @classmethod
    def _lookup_global_by_name(cls, name: str) -> object:
        """
        Like `globals()[name]` but supports dotted names.
        """
    @staticmethod
    def _getattr_by_name(root: object, name: str) -> object:
        """
        Like `getattr(root, name)` but supports dotted names.
        """

class _OpStrPickleData(_OpPickleData):
    name: Incomplete
    def __init__(self, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> str: ...

class _OpOverloadPickleData(_OpPickleData):
    name: Incomplete
    def __init__(self, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverload: ...

class _OpOverloadPacketPickleData(_OpPickleData):
    name: Incomplete
    def __init__(self, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverloadPacket: ...

class _OpBuiltinPickleData(_OpPickleData):
    root: Incomplete
    name: Incomplete
    def __init__(self, root: str, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> object: ...

class _OpOperatorPickleData(_OpPickleData):
    name: Incomplete
    def __init__(self, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> object: ...

class _GraphPickleData:
    tracer_cls: Incomplete
    tracer_extras: Incomplete
    nodes: Incomplete
    def __init__(self, graph: torch.fx.Graph, options: Options) -> None: ...
    def unpickle(self, gm: torch.fx.GraphModule, unpickle_state: _UnpickleState) -> torch.fx.Graph: ...

class _TracingContextPickleData:
    @classmethod
    def reduce_helper(cls, pickler: GraphPickler, obj: torch._guards.TracingContext) -> tuple[Callable[[Self, _UnpickleState], torch._guards.TracingContext], tuple[Self, _UnpickleStateToken]]: ...
    module_context: Incomplete
    frame_summary_stack: Incomplete
    loc_in_frame: Incomplete
    aot_graph_name: Incomplete
    params_flat: Incomplete
    params_flat_unwrap_subclasses: Incomplete
    params_unwrapped_to_flat_index: Incomplete
    output_strides: Incomplete
    force_unspec_int_unbacked_size_like: Incomplete
    def __init__(self, context: TracingContext) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> TracingContext: ...
