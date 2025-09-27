import _thread
import contextlib
import pickle
import torch
import types
import weakref
from _typeshed import Incomplete
from collections.abc import Generator
from torch._C._distributed_rpc import PyRRef as PyRRef
from torch._awaits import _Await as _Await
from torch._sources import fake_range as fake_range, get_source_lines_and_file as get_source_lines_and_file, parse_def as parse_def
from torch.distributed.rpc import RRef as RRef
from torch.futures import Future as Future
from typing import Any, Callable, Final, TypeVar
from typing_extensions import ParamSpec

_P = ParamSpec('_P')
_R = TypeVar('_R')
IS_PY310_PLUS: Final[bool]
BuiltinUnionType: type | tuple[type, ...]
BuiltinUnionType = types.UnionType
LockType: type
LockType = _thread.LockType
boolean_dispatched: weakref.WeakKeyDictionary[Callable, dict[str, Callable]]
FAKE_FILENAME_PREFIX: str

def is_final(ann) -> bool: ...

class BroadcastingListCls:
    def __getitem__(self, types) -> None: ...

BroadcastingList1: Incomplete

def is_scripting() -> bool:
    """
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.
    .. testcode::

        import torch

        @torch.jit.unused
        def unsupported_linear_op(x):
            return x

        def linear(x):
            if torch.jit.is_scripting():
                return torch.linear(x)
            else:
                return unsupported_linear_op(x)
    """
def _qualified_name(obj, mangle_name: bool = True) -> str: ...

class SourceLoader:
    content: Incomplete
    def __init__(self) -> None: ...
    def cache(self, fn, source) -> None: ...
    def get_source(self, fn): ...

loader: Incomplete

def createResolutionCallbackFromEnv(lookup_base):
    """
    Creates a resolution callback that will look up qualified names in an
    environment, starting with `lookup_base` for the base of any qualified
    names, then proceeding down the lookup chain with the resolved object.

    You should not use this directly, it should only be used from the other
    createResolutionCallbackFrom* functions.
    """
def createResolutionCallbackFromFrame(frames_up: int = 0):
    '''
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallbackFromFrame (by default).

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallbackFromFrame. Also for example, if frames_up is set
    to 1, then the frame of the caller\'s caller of createResolutionCallbackFromFrame
    will be taken.

    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallbackFromFrame(1)
            print(cb("foo"))


        def baz():
            foo = 2
            bar()


        baz()
    '''
def get_closure(fn):
    """
    Get a dictionary of closed over variables from a function
    """
def createResolutionCallbackFromClosure(fn):
    """
    Create a resolutionCallback by introspecting the function instead of
    looking up the stack for the enclosing scope
    """
def can_compile_class(cls) -> bool: ...
def get_callable_argument_names(fn) -> list[str]:
    """
    Gets names of all POSITIONAL_OR_KEYWORD arguments for callable `fn`.
    Returns an empty list when other types of arguments are present.

    This is used by `torch.jit.trace` to assign meaningful argument names to
    traced functions and modules.

    Args:
        fn: A callable.
    Returns:
        Argument names: List[str]
    """
def get_annotation_str(annotation):
    """
    Convert an AST node containing a type annotation to the string present in the source
    that represents the same annotation.
    """
def get_type_hint_captures(fn):
    """
    Get a dictionary containing type resolution mappings necessary to resolve types
    for the literal annotations on 'fn'. These are not considered to be closed-over by fn
    and must be obtained separately (e.g. using this function).

    Args:
        fn: A callable.
    Returns:
        A Dict[str, Any] containing a mapping from the literal annotations used on
        fn to the Python objects they refer to.
    """
def createResolutionCallbackForClassMethods(cls):
    """
    This looks at all the methods defined in a class and pulls their closed-over
    variables into a dictionary and uses that to resolve variables.
    """
def boolean_dispatch(arg_name, arg_index, default, if_true, if_false, module_name, func_name):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """

class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """
    UNUSED: str
    IGNORE: str
    EXPORT: str
    DEFAULT: str
    COPY_TO_SCRIPT_WRAPPER: str
    _DROP: str

def export(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a
    :class:`ScriptModule` and should be compiled.

    ``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.
    Functions and methods called from ``forward`` are compiled as they are seen
    by the compiler, so they do not need this decorator either.

    Example (using ``@torch.jit.export`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def implicitly_compiled_method(self, x):
                return x + 99

            # `forward` is implicitly decorated with `@torch.jit.export`,
            # so adding it here would have no effect
            def forward(self, x):
                return x + 10

            @torch.jit.export
            def another_forward(self, x):
                # When the compiler sees this call, it will compile
                # `implicitly_compiled_method`
                return self.implicitly_compiled_method(x)

            def unused_method(self, x):
                return x - 20

        # `m` will contain compiled methods:
        #     `forward`
        #     `another_forward`
        #     `implicitly_compiled_method`
        # `unused_method` will not be compiled since it was not called from
        # any compiled methods and wasn't decorated with `@torch.jit.export`
        m = torch.jit.script(MyModule())
    """
def unused(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    '''
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self, use_memory_efficient):
                    super().__init__()
                    self.use_memory_efficient = use_memory_efficient

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb

                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10


            m = torch.jit.script(MyModule(use_memory_efficient=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    '''

class _IgnoreContextManager(contextlib.AbstractContextManager):
    def __init__(self, **kwargs) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

def ignore(drop: bool = False, **kwargs):
    '''
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. If called from TorchScript,
    ignored functions will dispatch the call to the Python interpreter. Models with ignored
    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn


        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb

                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x


        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save("m.pt")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save("m.pt")

    .. testcleanup::

        import os
        os.remove(\'m.pt\')
    '''
def _drop(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
def _copy_to_script_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
def module_has_exports(mod): ...
def should_drop(fn) -> bool: ...
def is_ignored_fn(fn) -> bool: ...
def _is_drop_fn(fn) -> bool: ...
def is_static_fn(cls, fn) -> bool: ...
def get_static_fn(cls, fn): ...
def get_torchscript_modifier(fn): ...
def copy_torchscript_modifier(orig, new) -> None: ...

_overloaded_fns: dict[str, list[Callable]]
_OVERLOAD_EXAMPLE: str

def get_overload_no_implementation_error_message(kind, obj): ...
def _check_overload_body(func): ...
def _overload(func): ...
def _get_fn_overloads(qual_name): ...
def _clear_fn_overloads(qual_name) -> None: ...
def get_class_name_lineno(method) -> tuple[str, int]: ...

_overloaded_methods: dict[str, dict[str, list[Callable]]]
_overloaded_method_class_fileno: dict[tuple[str, str], int]

def _overload_method(func): ...
def _get_overloaded_methods(method, mod_class): ...
def is_tuple(ann) -> bool: ...
def is_list(ann) -> bool: ...
def is_dict(ann) -> bool: ...
def is_union(ann): ...
def is_optional(ann): ...
def is_future(ann) -> bool: ...
def is_await(ann) -> bool: ...
def is_rref(ann) -> bool: ...
def is_rref_instance(obj) -> bool: ...
def _try_get_dispatched_fn(fn): ...
def _get_named_tuple_properties(obj, loc: torch._C._jit_tree_views.SourceRange | None = None, rcb=None): ...
def _create_named_tuple(t, unqual_name: str, field_names: list[str], defaults: tuple[Any, ...]): ...
@contextlib.contextmanager
def _disable_emit_hooks() -> Generator[None]: ...
def _disable_emit_hooks_decorator(_DecoratorContextManager) -> None: ...
def _is_exception(obj) -> bool: ...
def raise_error_container_parameter_missing(target_type) -> None: ...

_RAW_TYPE_NAME_MAPPING: Incomplete

def check_args_exist(target_type) -> None: ...
def check_empty_containers(obj) -> None: ...
def container_checker(obj, target_type) -> bool: ...
def _isinstance(obj, target_type) -> bool: ...

class _TensorExtractor(pickle.Pickler):
    tensors: Incomplete
    def __init__(self, *args, tensors: list[torch.Tensor], **kwargs) -> None: ...
    def persistent_id(self, obj): ...

def _extract_tensors(obj):
    """
    This function is exclusively called from C++.
    See ``torch/csrc/jit/python/python_ivalue.h``.

    It extracts the tensors contained in the given object, through pickling.
    """
def _get_model_id(obj) -> str | None: ...
