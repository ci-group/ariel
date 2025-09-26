import torch
from ._serialization import validate_map_location as validate_map_location
from _typeshed import Incomplete
from torch._classes import classes as classes
from torch._jit_internal import _get_model_id as _get_model_id, _qualified_name as _qualified_name
from torch._utils_internal import log_torchscript_usage as log_torchscript_usage
from torch.jit._builtins import _register_builtin as _register_builtin
from torch.jit._fuser import _graph_for as _graph_for, _script_method_graph_for as _script_method_graph_for
from torch.jit._monkeytype_config import JitTypeTraceConfig as JitTypeTraceConfig, JitTypeTraceStore as JitTypeTraceStore, monkeytype_trace as monkeytype_trace
from torch.jit._recursive import ScriptMethodStub as ScriptMethodStub, _compile_and_register_class as _compile_and_register_class, infer_methods_to_compile as infer_methods_to_compile, wrap_cpp_module as wrap_cpp_module
from torch.jit._state import _enabled as _enabled, _set_jit_function_cache as _set_jit_function_cache, _set_jit_overload_cache as _set_jit_overload_cache, _try_get_jit_cached_function as _try_get_jit_cached_function, _try_get_jit_cached_overloads as _try_get_jit_cached_overloads
from torch.jit.frontend import get_default_args as get_default_args, get_jit_class_def as get_jit_class_def, get_jit_def as get_jit_def
from torch.nn import Module as Module
from torch.overrides import has_torch_function as has_torch_function, has_torch_function_unary as has_torch_function_unary, has_torch_function_variadic as has_torch_function_variadic
from torch.package import PackageExporter as PackageExporter, PackageImporter as PackageImporter
from torch.utils import set_module as set_module
from typing import Any, Callable, NamedTuple

type_trace_db: Incomplete
ScriptFunction = torch._C.ScriptFunction

def _reduce(cls) -> None: ...

class Attribute(NamedTuple):
    value: Incomplete
    type: Incomplete

def Attribute(value, type): ...
def _get_type_trace_db(): ...
def _get_function_from_type(cls, name): ...
def _is_new_style_class(cls): ...

class OrderedDictWrapper:
    _c: Incomplete
    def __init__(self, _c) -> None: ...
    def keys(self): ...
    def values(self): ...
    def __len__(self) -> int: ...
    def __delitem__(self, k) -> None: ...
    def items(self): ...
    def __setitem__(self, k, v) -> None: ...
    def __contains__(self, k) -> bool: ...
    def __getitem__(self, k): ...

class OrderedModuleDict(OrderedDictWrapper):
    _python_modules: Incomplete
    def __init__(self, module, python_dict) -> None: ...
    def items(self): ...
    def __contains__(self, k) -> bool: ...
    def __setitem__(self, k, v) -> None: ...
    def __getitem__(self, k): ...

class ScriptMeta(type):
    def __init__(cls, name, bases, attrs) -> None: ...

class _CachedForward:
    def __get__(self, obj, cls): ...

class ScriptWarning(Warning): ...

def script_method(fn): ...

class ConstMap:
    const_mapping: Incomplete
    def __init__(self, const_mapping) -> None: ...
    def __getattr__(self, attr): ...

def unpackage_script_module(importer: PackageImporter, script_module_id: str) -> torch.nn.Module:
    """
    Call by ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function.

    Performs work of loading and returning a ScriptModule from a ``torch.package`` archive.
    """

_magic_methods: Incomplete

class RecursiveScriptClass:
    """Wrapper for a TorchScript class instance for use in Python.

        An analogue of RecursiveScriptModule for regular objects that are not modules.
        This class is a wrapper around a torch._C.ScriptObject that represents an instance
        of a TorchScript class and allows it to be used in Python.

        Attributes:
            _c [torch._C.ScriptObject]: The C++ object to which attribute lookups and method
                calls are forwarded.
            _props [Dict[str, property]]: A dictionary of properties fetched from self._c and
                exposed on this wrppaer.
        """
    _c: Incomplete
    _props: Incomplete
    def __init__(self, cpp_class) -> None: ...
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value): ...
    def forward_magic_method(self, method_name, *args, **kwargs): ...
    def __getstate__(self) -> None: ...
    def __iadd__(self, other): ...

def method_template(self, *args, **kwargs): ...

class ScriptModule(Module, metaclass=ScriptMeta):
    """Wrapper for C++ torch::jit::Module with methods, attributes, and parameters.

        A wrapper around C++ ``torch::jit::Module``. ``ScriptModule``\\s
        contain methods, attributes, parameters, and
        constants. These can be accessed the same way as on a normal ``nn.Module``.
        """
    __jit_unused_properties__: Incomplete
    def __init__(self) -> None: ...
    forward: Callable[..., Any]
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value): ...
    def define(self, src): ...
    def _replicate_for_data_parallel(self): ...
    def __reduce_package__(self, exporter: PackageExporter):
        """Save a ScriptModule inside of a ``torch.package`` archive.

            Called by ``torch.package.PackageExporter``'s Pickler's ``persistent_id`` when
            saving TorchScript objects. Performs act of saving a ScriptModule inside of
            a ``torch.package`` archive.

            Returns method to load the ScriptModule from a ``torch.package.PackageImporter``'s
            Pickler's ``persistent_load`` function.
            """

class RecursiveScriptModule(ScriptModule):
    """Retain the existing isinstance(ScriptModule) behavior.

        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's ``nn.Module`` and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ``ScriptModule`` can
        have submodules, parameters, and methods. In ``nn.Module``\\s methods are implemented
        as Python functions, but in ``ScriptModule``\\s methods are implemented as
        TorchScript functions, a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ``ScriptModule``\\s code to run without the need for a Python interpreter.

        ``ScriptModule``\\s should not be created manually, instead use
        either :func:`tracing <torch.jit.trace>` or :func:`scripting <torch.jit.script>`.
        Tracing and scripting can be applied incrementally and :ref:`composed as necessary <Types>`.

        * Tracing records the tensor operations as executed with a set of example inputs and uses these
          operations to construct a computation graph. You can use the full dynamic behavior of Python with tracing,
          but values other than Tensors and control flow aren't captured in the graph.

        * Scripting inspects the Python code of the model
          and compiles it to TorchScript. Scripting allows the use of many `types`_ of values and supports dynamic control flow.
          Many, but not all features of Python are supported by the compiler, so changes to the source code may be necessary.
        """
    _disable_script_meta: bool
    _c: Incomplete
    def __init__(self, cpp_module) -> None: ...
    @staticmethod
    def _construct(cpp_module, init_fn):
        """
            Construct a RecursiveScriptModule that's ready for use.

            PyTorch code should use this to construct a RecursiveScriptModule instead
            of instead of calling `__init__` directly, as it makes sure the
            object is properly finalized (and in the future, we may take
            control of how the RecursiveScriptModule instance is created).

            Args:
                cpp_module:  The C++ Module that will hold the actual state of
                             this RecursiveScriptModule instance.
                init_fn:  Lambda that initializes the RecursiveScriptModule passed to it.
            """
    @staticmethod
    def _finalize_scriptmodule(script_module) -> None: ...
    _concrete_type: Incomplete
    _modules: Incomplete
    _parameters: Incomplete
    _buffers: Incomplete
    __dict__: Incomplete
    def _reconstruct(self, cpp_module) -> None:
        """
            Re-construct an instance of RecursiveScriptModule using an instance of a C++ module.

            Args:
                cpp_module: The C++ module that this RecursiveScriptModule will be rebuilt around.
            """
    @property
    def graph(self):
        """Return a string representation of the internal graph for the ``forward`` method.

            See :ref:`interpreting-graphs` for details.
            """
    @property
    def inlined_graph(self):
        """
            Return a string representation of the internal graph for the ``forward`` method.

            This graph will be preprocessed to inline all function and method calls.
            See :ref:`interpreting-graphs` for details.
            """
    @property
    def code(self):
        """
            Return a pretty-printed representation (as valid Python syntax) of the internal graph for the ``forward`` method.

            See :ref:`inspecting-code` for details.
            """
    @property
    def code_with_constants(self):
        """Return a tuple.

            Returns a tuple of:

            [0] a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See `code`.
            [1] a ConstMap following the CONSTANT.cN format of the output in [0].
            The indices in the [0] output are keys to the underlying constant's values.

            See :ref:`inspecting-code` for details.
            """
    def save(self, f, **kwargs):
        """Save with a file-like object.

            save(f, _extra_files={})

            See :func:`torch.jit.save <torch.jit.save>` which accepts a file-like object.
            This function, torch.save(), converts the object to a string, treating it as a path.
            DO NOT confuse these two functions when it comes to the 'f' parameter functionality.
            """
    def _save_for_lite_interpreter(self, *args, **kwargs):
        """Add (or update) the bytecode session to the script model.

            _save_for_lite_interpreter(f)

            The updated model is used
            in lite interpreter for mobile applications.

            Args:
                f: a string containing a file name.
                _extra_files: Map from filename to contents which will be stored as part of 'f'.

            """
    def _save_to_buffer_for_lite_interpreter(self, *args, **kwargs): ...
    def save_to_buffer(self, *args, **kwargs): ...
    def get_debug_state(self, *args, **kwargs): ...
    def extra_repr(self): ...
    def graph_for(self, *args, **kwargs): ...
    @property
    def original_name(self): ...
    def define(self, src) -> None: ...
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value): ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def forward_magic_method(self, method_name, *args, **kwargs): ...
    def __iter__(self): ...
    def __getitem__(self, idx): ...
    def __len__(self) -> int: ...
    def __contains__(self, key) -> bool: ...
    def __dir__(self): ...
    def __bool__(self) -> bool: ...
    def _replicate_for_data_parallel(self): ...

def _get_methods(cls): ...

_compiled_methods_allowlist: Incomplete

def _make_fail(name): ...

class RecursiveScriptClass: ...

class ScriptModule(torch.nn.Module):
    def __init__(self, arg=None) -> None: ...

class RecursiveScriptModule(ScriptModule):
    def __init__(self, arg=None) -> None: ...

def call_prepare_scriptable_func_impl(obj, memo): ...
def call_prepare_scriptable_func(obj): ...
def create_script_dict(obj):
    """
    Create a ``torch._C.ScriptDict`` instance with the data from ``obj``.

    Args:
        obj (dict): The Python dictionary that is used to initialize the ``ScriptDict``
                    returned by this function.

    Returns:
        An instance of ``torch._C.ScriptDict`` that has the same data as ``obj``
        and can be passed between Python and TorchScript with reference semantics and
        zero copy overhead.
    """
def create_script_list(obj, type_hint=None):
    """
    Create a ``torch._C.ScriptList`` instance with the data from ``obj``.

    Args:
        obj (dict): The Python list that is used to initialize the ``ScriptList``
                    returned by this function.
    Returns:
        An instance of ``torch._C.ScriptList`` that has the same data as ``obj``
        and can be passed between Python and TorchScript with reference semantics and
        zero copy overhead.
    """

_TOPLEVEL: bool

def _script_impl(obj, optimize=None, _frames_up: int = 0, _rcb=None, example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None): ...
def script(obj, optimize=None, _frames_up: int = 0, _rcb=None, example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None):
    """Script the function.

    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    :ref:`language-reference`.

    Scripting a dictionary or list copies the data inside it into a TorchScript instance than can be
    subsequently passed by reference between Python and TorchScript with zero copy overhead.

    ``torch.jit.script`` can be used as a function for modules, functions, dictionaries and lists
     and as a decorator ``@torch.jit.script`` for :ref:`torchscript-classes` and functions.

    Args:
        obj (Callable, class, or nn.Module):  The ``nn.Module``, function, class type,
                                                  dictionary, or list to compile.
        example_inputs (Union[List[Tuple], Dict[Callable, List[Tuple]], None]): Provide example inputs
            to annotate the arguments for a function or ``nn.Module``.

    Returns:
        If ``obj`` is ``nn.Module``, ``script`` returns
        a :class:`ScriptModule` object. The returned :class:`ScriptModule` will
        have the same set of sub-modules and parameters as the
        original ``nn.Module``. If ``obj`` is a standalone function,
        a :class:`ScriptFunction` will be returned. If ``obj`` is a ``dict``, then
        ``script`` returns an instance of `torch._C.ScriptDict`. If ``obj`` is a ``list``,
        then ``script`` returns an instance of `torch._C.ScriptList`.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

            print(type(foo))  # torch.jit.ScriptFunction

            # See the compiled graph as Python code
            print(foo.code)

            # Call the function using the TorchScript interpreter
            foo(torch.ones(2, 2), torch.ones(2, 2))

        .. testoutput::
            :hide:

            ...

    ****Scripting a function using example_inputs**
        Example inputs can be used to annotate a function arguments.

        Example (annotating a function before scripting):

        .. testcode::

            import torch

            def test_sum(a, b):
                return a + b

            # Annotate the arguments to be int
            scripted_fn = torch.jit.script(test_sum, example_inputs=[(3, 4)])

            print(type(scripted_fn))  # torch.jit.ScriptFunction

            # See the compiled graph as Python code
            print(scripted_fn.code)

            # Call the function using the TorchScript interpreter
            scripted_fn(20, 100)

        .. testoutput::
            :hide:

            ...

    **Scripting an nn.Module**
        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively
        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses
        features supported in TorchScript, no changes to the original module code should be necessary. ``script``
        will construct :class:`ScriptModule` that has copies of the attributes, parameters, and methods of
        the original module.

        Example (scripting a simple module with a Parameter):

        .. testcode::

            import torch

            class MyModule(torch.nn.Module):
                def __init__(self, N, M):
                    super().__init__()
                    # This parameter will be copied to the new ScriptModule
                    self.weight = torch.nn.Parameter(torch.rand(N, M))

                    # When this submodule is used, it will be compiled
                    self.linear = torch.nn.Linear(N, M)

                def forward(self, input):
                    output = self.weight.mv(input)

                    # This calls the `forward` method of the `nn.Linear` module, which will
                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
                    output = self.linear(output)
                    return output

            scripted_module = torch.jit.script(MyModule(2, 3))

        Example (scripting a module with traced submodules):

        .. testcode::

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MyModule(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    # torch.jit.trace produces a ScriptModule's conv1 and conv2
                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                def forward(self, input):
                    input = F.relu(self.conv1(input))
                    input = F.relu(self.conv2(input))
                    return input

            scripted_module = torch.jit.script(MyModule())

        To compile a method other than ``forward`` (and recursively compile anything it calls), add
        the :func:`@torch.jit.export <torch.jit.export>` decorator to the method. To opt out of compilation
        use :func:`@torch.jit.ignore <torch.jit.ignore>` or :func:`@torch.jit.unused <torch.jit.unused>`.

        Example (an exported and ignored method in a module)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                @torch.jit.export
                def some_entry_point(self, input):
                    return input + 10

                @torch.jit.ignore
                def python_only_fn(self, input):
                    # This function won't be compiled, so any
                    # Python APIs can be used
                    import pdb

                    pdb.set_trace()

                def forward(self, input):
                    if self.training:
                        self.python_only_fn(input)
                    return input * 99


            scripted_module = torch.jit.script(MyModule())
            print(scripted_module.some_entry_point(torch.randn(2, 2)))
            print(scripted_module(torch.randn(2, 2)))

        Example ( Annotating forward of nn.Module using example_inputs)::

            import torch
            import torch.nn as nn
            from typing import NamedTuple

            class MyModule(NamedTuple):
            result: List[int]

            class TestNNModule(torch.nn.Module):
                def forward(self, a) -> MyModule:
                    result = MyModule(result=a)
                    return result

            pdt_model = TestNNModule()

            # Runs the pdt_model in eager model with the inputs provided and annotates the arguments of forward
            scripted_model = torch.jit.script(pdt_model, example_inputs={pdt_model: [([10, 20, ], ), ], })

            # Run the scripted_model with actual inputs
            print(scripted_model([20]))
    """
def _check_overload_defaults(impl_defaults, overload_defaults, loc) -> None: ...
def _compile_function_with_overload(overload_fn, qual_name, impl_fn): ...
def _get_overloads(obj): ...
def _check_directly_compile_overloaded(obj) -> None: ...
def interface(obj):
    '''Decorate to annotate classes or modules of different types.

    This decorator can be used to define an interface that can be used to annotate
    classes or modules of different types. This can be used for to annotate a submodule
    or attribute class that could have different types that implement the same
    interface, or which could be swapped at runtime; or to store a list of modules or
    classes of varying types.

    It is sometimes used to implement "Callables" - functions or modules that implement
    an interface but whose implementations differ and which can be swapped out.

    Example:
    .. testcode::

        import torch
        from typing import List

        @torch.jit.interface
        class InterfaceType:
            def run(self, x: torch.Tensor) -> torch.Tensor:
                pass

        # implements InterfaceType
        @torch.jit.script
        class Impl1:
            def run(self, x: torch.Tensor) -> torch.Tensor:
                return x.relu()

        class Impl2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.val = torch.rand(())

            @torch.jit.export
            def run(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.val

        def user_fn(impls: List[InterfaceType], idx: int, val: torch.Tensor) -> torch.Tensor:
            return impls[idx].run(val)

        user_fn_jit = torch.jit.script(user_fn)

        impls = [Impl1(), torch.jit.script(Impl2())]
        val = torch.rand(4, 4)
        user_fn_jit(impls, 0, val)
        user_fn_jit(impls, 1, val)
    '''
def _recursive_compile_class(obj, loc): ...
CompilationUnit = torch._C.CompilationUnit

def pad(s: str, padding: int, offset: int = 0, char: str = ' '): ...

class _ScriptProfileColumn:
    header: Incomplete
    alignment: Incomplete
    offset: Incomplete
    rows: dict[int, Any]
    def __init__(self, header: str, alignment: int = 4, offset: int = 0) -> None: ...
    def add_row(self, lineno: int, value: Any): ...
    def materialize(self): ...

class _ScriptProfileTable:
    cols: Incomplete
    source_range: Incomplete
    def __init__(self, cols: list[_ScriptProfileColumn], source_range: list[int]) -> None: ...
    def dump_string(self): ...

class _ScriptProfile:
    profile: Incomplete
    def __init__(self) -> None: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def dump_string(self) -> str: ...
    def dump(self) -> None: ...

def _unwrap_optional(x): ...
