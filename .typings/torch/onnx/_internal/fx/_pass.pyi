import abc
import contextlib
import dataclasses
import torch.fx
from collections.abc import Generator
from torch._subclasses import fake_tensor as fake_tensor
from torch._subclasses.fake_tensor import unset_fake_temporarily as unset_fake_temporarily
from typing import Any, Callable

@dataclasses.dataclass
class PackageInfo:
    package_name: str
    version: str | None
    commit_hash: str | None
    def to_onnx_domain_string(self) -> str: ...
    @classmethod
    def from_python_class(cls, python_class_name: type | str) -> PackageInfo: ...

@dataclasses.dataclass
class GraphModuleOnnxMeta:
    package_info: PackageInfo

@contextlib.contextmanager
def _patch_difflib_sequence_matcher_init() -> Generator[None]:
    """Context patching `difflib.SequenceMatcher` for fx readable graph.

    Under this context, the `autojunk` argument of `difflib.SequenceMatcher` will always
    be considered as `False`. This is to prevent `difflib.SequenceMatcher` recognizing
    stacktrace messages in fx readable graph as junk, as these messages tend to be long (>200)
    and repeat multiple times, which falls under the junk filter criteria.

    `difflib.SequenceMatcher` is used underneath by all sorts of diffing functions
    in `difflib`, including `difflib.unified_diff`, `difflib.ndiff`, `difflib.context_diff`.
    Unfortunately, there is no way to pass `autojunk` argument to these functions, and
    they all default to `True`. This context patching will affect all of them.

    `Reference: Automatic junk heuristic <https://docs.python.org/3/library/difflib.html>`_
    """
def _unified_diff(a: str, b: str) -> str:
    '''Return a string containing the unified diff of two strings.

    This function calls a patched version of `difflib.unified_diff` with `autojunk` set
    to `False` for `difflib.SequenceMatcher` class. More details can be found in
    `_patch_difflib_sequence_matcher_init` function.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        The unified diff of the two strings. If there is no diff, return "<no diff>".

    Example::

        >>> a = \'\'\'class GraphModule(torch.nn.Module):
        ...     def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view = input_ids.view(-1, 3);  input_ids = None
        ... \'\'\'
        >>> b = \'\'\'class <lambda>(torch.nn.Module):
        ...     def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
        ... \'\'\'
        >>> print(_unified_diff(a, b))
        ---
        +++
        @@ -1,4 +1,4 @@
        -class GraphModule(torch.nn.Module):
        -    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        +class <lambda>(torch.nn.Module):
        +    def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
                # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        -        view = input_ids.view(-1, 3);  input_ids = None
        +        view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
    '''
def _transform_diagnose_call_message_formatter(run: Callable, self: Transform, *args: Any, **kwargs: Any) -> str: ...
def maybe_fx_graph_tabular(graph: torch.fx.Graph) -> str | None:
    """Return the Graph nodes in tabular format. Equivalent to stdout of `graph.print_tabular()`.
    If `tabulate` is not installed, return `None`.

    Args:
        graph: The Graph to print.

    Returns:
        The Graph printed in a tabular format. None if `tabulate` is not installed.
    """

class Transform(abc.ABC, metaclass=abc.ABCMeta):
    """Base class for FX graph transformations to be used by FX-ONNX exporter.

    Similar to `FX Interpreter <https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter>`_,
    specializations of this class execute the FX graph Node-by-Node.
    Methods in the `Transform` class can be overridden to customize the behavior of the model.
    This pattern can be useful for many things, including writing code transformations as well as analysis passes.

    The following methods can be overridden::

        _run()
            +-- run_node()
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    One important aspect to note is that if the transformation modifies the model input and/or output signature,
    (e.g. additional inputs/outputs are added to the model), :class:`InputAdaptStep` and/or :class:`OutputAdaptStep`
    are needed to reconcile :attr:`ONNXProgram.model_proto`.
    That is, the model signature and the model representation must match.

    TODO(bowbao): Add more overridable methods in call hierarchy
    TODO(bowbao): Create an example once more overridable methods are added.
    """
    module: torch.fx.GraphModule
    fake_mode: fake_tensor.FakeTensorMode | None
    def __init__(self, module: torch.fx.GraphModule) -> None:
        """Initialize the transform.

        Args:
            module: The module to be transformed.
        """
    def _detect_fake_mode(self) -> fake_tensor.FakeTensorMode | None:
        """Detect fake mode from the graph.

        Scan through all nodes in graph and their meta['val'] to detect fake mode.
        """
    def _maybe_fakefy_args(self, fake_mode: fake_tensor.FakeTensorMode | None, *args: Any) -> tuple[Any, ...]: ...
    @abc.abstractmethod
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule: ...
    def run(self, *args, **kwargs) -> torch.fx.GraphModule:
        """Run the transform on `self.module`.

        Note that this method may or may not mutate `self.module`, and the returned
        `GraphModule` could be either `self.module` or a new `GraphModule`.

        Args:
            *args: Positional arguments for `self.module` to run.
            **kwargs: Keyword arguments for `self.module` to run.
        """
