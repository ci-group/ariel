import torch._C
from contextlib import contextmanager
from torch._jit_internal import Future as Future, export as export, ignore as ignore, unused as unused
from torch.jit._async import fork as fork, wait as wait
from torch.jit._freeze import freeze as freeze, optimize_for_inference as optimize_for_inference
from torch.jit._fuser import set_fusion_strategy as set_fusion_strategy
from torch.jit._script import Attribute as Attribute, CompilationUnit as CompilationUnit, ScriptFunction as ScriptFunction, ScriptModule as ScriptModule, interface as interface, script as script
from torch.jit._serialization import load as load, save as save
from torch.jit._trace import trace as trace, trace_module as trace_module
from typing import Any

__all__ = ['Attribute', 'CompilationUnit', 'Error', 'Future', 'ScriptFunction', 'ScriptModule', 'annotate', 'enable_onednn_fusion', 'export', 'export_opnames', 'fork', 'freeze', 'interface', 'ignore', 'isinstance', 'load', 'onednn_fusion_enabled', 'optimize_for_inference', 'save', 'script', 'script_if_tracing', 'set_fusion_strategy', 'strict_fusion', 'trace', 'trace_module', 'unused', 'wait']

_fork = fork
_wait = wait
_set_fusion_strategy = set_fusion_strategy

def export_opnames(m):
    """
    Generate new bytecode for a Script module.

    Returns what the op list would be for a Script Module based off the current code base.

    If you have a LiteScriptModule and want to get the currently present
    list of ops call _export_operator_list instead.
    """
Error = torch._C.JITException

def annotate(the_type, the_value):
    '''Use to give type of `the_value` in TorchScript compiler.

    This method is a pass-through function that returns `the_value`, used to hint TorchScript
    compiler the type of `the_value`. It is a no-op when running outside of TorchScript.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume
      it is type `T` rather than `Optional[T]`

    Note that `annotate()` does not help in `__init__` method of `torch.nn.Module` subclasses because it
    is executed in eager mode. To annotate types of `torch.nn.Module` attributes,
    use :meth:`~torch.jit.Attribute` instead.

    Example:

    .. testcode::

        import torch
        from typing import Dict

        @torch.jit.script
        def fn():
            # Telling TorchScript that this empty dictionary is a (str -> int) dictionary
            # instead of default dictionary type of (str -> Tensor).
            d = torch.jit.annotate(Dict[str, int], {})

            # Without `torch.jit.annotate` above, following statement would fail because of
            # type mismatch.
            d["name"] = 20

    .. testcleanup::

        del fn

    Args:
        the_type: Python type that should be passed to TorchScript compiler as type hint for `the_value`
        the_value: Value or expression to hint type for.

    Returns:
        `the_value` is passed back as return value.
    '''
def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing.

    ``torch.jit.script`` has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """
def isinstance(obj, target_type):
    '''
    Provide container type refinement in TorchScript.

    It can refine parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``,
    ``Dict[str, List[torch.Tensor]]``, ``Optional[Tuple[int,str,int]]``. It can also
    refine basic types such as bools and ints that are available in TorchScript.

    Args:
        obj: object to refine the type of
        target_type: type to try to refine obj to
    Returns:
        ``bool``: True if obj was successfully refined to the type of target_type,
            False otherwise with no new type refinement


    Example (using ``torch.jit.isinstance`` for type refinement):
    .. testcode::

        import torch
        from typing import Any, Dict, List

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input: Any): # note the Any type
                if torch.jit.isinstance(input, List[torch.Tensor]):
                    for t in input:
                        y = t.clamp(0, 0.5)
                elif torch.jit.isinstance(input, Dict[str, str]):
                    for val in input.values():
                        print(val)

        m = torch.jit.script(MyModule())
        x = [torch.rand(3,3), torch.rand(4,3)]
        m(x)
        y = {"key1":"val1","key2":"val2"}
        m(y)
    '''

class strict_fusion:
    """
    Give errors if not all nodes have been fused in inference, or symbolically differentiated in training.

    Example:
    Forcing fusion of additions.

    .. code-block:: python

        @torch.jit.script
        def foo(x):
            with torch.jit.strict_fusion():
                return x + x + x

    """
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, tb: Any) -> None: ...

def enable_onednn_fusion(enabled: bool):
    """Enable or disables onednn JIT fusion based on the parameter `enabled`."""
def onednn_fusion_enabled():
    """Return whether onednn JIT fusion is enabled."""
