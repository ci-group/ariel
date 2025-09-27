from . import config as config
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._dynamo.utils import dynamo_timed as dynamo_timed
from torch._inductor.cudagraph_utils import BoxedDeviceIndex as BoxedDeviceIndex
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir as temporary_cache_dir
from torch._inductor.utils import BoxedBool as BoxedBool, InputType as InputType
from torch._subclasses import FakeTensorMode as FakeTensorMode
from torch.compiler._cache import CacheInfo as CacheInfo
from torch.fx import GraphModule as GraphModule
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv
from typing import Any, Callable, Literal

log: Incomplete

class CompiledArtifact:
    """
    CompiledArtifact class represents the precompiled inductor artifact that
    can be invoked in order to avoid repeated compilation.

    CompiledArtifact can be obtained by calling standalone_compile(gm, example_inputs)
    to create a fresh CompiledArtifact from a GraphModule and example inputs.

    Later this CompiledArtifact can be saved to disk, either as a binary or unpacked
    into the provided folder via the CompiledArtifact.save function.

    CompiledArtifact.load provides a way to create a CompiledArtifact from the
    binary or unpacked data.

    Finally, the CompiledArtifact can be invoked via the __call__ method
    to execute the precompiled artifact.
    """
    _compiled_fn: Callable[..., Any]
    _artifacts: tuple[bytes, CacheInfo] | None
    def __init__(self, compiled_fn: Callable[..., Any], artifacts: tuple[bytes, CacheInfo] | None) -> None: ...
    def __call__(self, *args: Any) -> Any: ...
    def save(self, *, path: str, format: Literal['binary', 'unpacked'] = 'binary') -> None: ...
    @staticmethod
    def load(*, path: str, format: Literal['binary', 'unpacked'] = 'binary') -> CompiledArtifact: ...

def standalone_compile(gm: GraphModule, example_inputs: Sequence[InputType], *, dynamic_shapes: Any, options: Any) -> CompiledArtifact: ...
