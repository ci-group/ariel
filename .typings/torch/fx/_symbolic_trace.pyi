import contextlib
import torch
import torch.utils._pytree as pytree
import types
from ._compatibility import compatibility as compatibility
from ._lazy_graph_module import _make_graph_module as _make_graph_module
from .graph import Graph as Graph, _PyTreeCodeGen as _PyTreeCodeGen, _PyTreeInfo as _PyTreeInfo
from .graph_module import GraphModule as GraphModule
from .node import Argument as Argument, base_types as base_types, map_aggregate as map_aggregate
from .proxy import ParameterProxy as ParameterProxy, Proxy as Proxy, Scope as Scope, ScopeContextManager as ScopeContextManager, TracerBase as TracerBase
from _typeshed import Incomplete
from collections.abc import Generator
from torch._C import ScriptObject as ScriptObject
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from types import FunctionType, ModuleType
from typing import Any, Callable, NamedTuple
from typing_extensions import TypeAlias

HAS_VARSTUFF: Incomplete
_orig_module_call: Callable
_orig_module_getattr: Callable
_proxyable_classes: dict[type, None]
_is_fx_tracing_flag: bool
_ConstantAttributeType: TypeAlias = torch.Tensor | torch.ScriptObject | FakeScriptObject | pytree.TreeSpec
_constant_attribute_types: Incomplete

def is_fx_tracing(): ...

class ProxyableClassMeta(type):
    """
    ProxyableClassMeta allows you to make construction of a given Python class
    symbolically traceable. For example::

        import torch
        import torch.fx


        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            def __init__(self, left, right):
                self.left, self.right = left, right

            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)


        def use_tensor_pair_ctor(x: TensorPair, y: torch.Tensor):
            s = x.add(TensorPair(y, y))
            return s.mul(x)


        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)

        traced = torch.fx.symbolic_trace(use_tensor_pair_ctor)
        print(traced.code)
        '''
        def forward(self, x : __main___TensorPair, y : torch.Tensor):
            tensor_pair = __main___TensorPair(y, y);  y = None
            add = x.add(tensor_pair);  tensor_pair = None
            mul = add.mul(x);  add = x = None
            return mul
        '''

    From this example, we can see that construction of a class (``TensorPair``)
    defined with ``ProxyableClassMeta`` as metaclass can be recorded in symbolic
    tracing.
    """
    def __init__(cls, name, bases, attrs) -> None: ...
    def __call__(cls, *args, **kwargs): ...

def _patch_function(fn: FunctionType, nargs: int) -> FunctionType: ...

class PHBase:
    """
    Object representing an input placeholder to `concrete_args`
    """
    def __repr__(self) -> str: ...

PH: Incomplete

class PHWithMeta(PHBase):
    """
    Object representing an input placeholder to `concrete_args`
    """
    ph_key: Incomplete
    def __init__(self, ph_key: str | None = None) -> None: ...

def _transfer_attrs(fr, to) -> None: ...

class Tracer(TracerBase):
    """Tracer(autowrap_modules=(math,), autowrap_functions=())

    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.

    Tracer can be subclassed to override various behaviors of the tracing
    process. The different behaviors that can be overridden are described
    in the docstrings of the methods on this class.
    """
    _autowrap_function_ids: set[int]
    _autowrap_search: list[ModuleType]
    param_shapes_constant: Incomplete
    submodule_paths: dict[torch.nn.Module, str] | None
    root_module_name: str
    scope: Incomplete
    module_stack: Incomplete
    num_calls: dict[str, int]
    node_name_to_scope: dict[str, tuple[str, type]]
    def __init__(self, autowrap_modules: tuple[ModuleType] = ..., autowrap_functions: tuple[Callable, ...] = (), param_shapes_constant: bool = False) -> None:
        """
        Construct a Tracer object.

        Args:

            autowrap_modules (Tuple[ModuleType]): defaults to `(math, )`,
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap(). Backward-compatibility for
                this parameter is guaranteed.

            autowrap_functions (Tuple[Callable, ...]): defaults to `()`,
                Python functions that should be wrapped automatically without
                needing to use fx.wrap(). Backward compatibility for this
                parameter is guaranteed.

            param_shapes_constant (bool): When this flag is set,  calls to shape,
                size and a few other shape like attributes of a module's parameter
                will be evaluated directly, rather than returning a new Proxy value
                for an attribute access. Backward compatibility for this parameter
                is guaranteed.
        """
    _qualname_counter: dict[str, int]
    def get_fresh_qualname(self, prefix: str) -> str:
        """
        Gets a fresh name for a prefix and returns it. This function ensures
        that it will not clash with an existing attribute on the graph.
        """
    def create_arg(self, a: Any) -> Argument:
        """
        A method to specify the behavior of tracing when preparing values to
        be used as arguments to nodes in the ``Graph``.

        By default, the behavior includes:

        #. Iterate through collection types (e.g. tuple, list, dict) and recursively
           call ``create_args`` on the elements.
        #. Given a Proxy object, return a reference to the underlying IR ``Node``
        #. Given a non-Proxy Tensor object, emit IR for various cases:

            * For a Parameter, emit a ``get_attr`` node referring to that Parameter
            * For a non-Parameter Tensor, store the Tensor away in a special
              attribute referring to that attribute.

        This method can be overridden to support more types.

        Args:

            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.


        Returns:

            The value ``a`` converted into the appropriate ``Argument``
        """
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        '''
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        '''
    def path_of_module(self, mod: torch.nn.Module) -> str:
        '''
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".

        Args:

            mod (str): The ``Module`` to retrieve the qualified name for.
        '''
    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:

            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: dict[str, Any]):
        """
        Method that specifies the behavior of this ``Tracer`` when we call getattr
        on a call to an ``nn.Module`` instance.

        By default, the behavior is to return a proxy value for the attribute. It
        also stores the proxy value in the ``parameter_proxy_cache``, so that future
        calls will reuse the proxy rather than creating a new one.

        This method can be overridden to --for example-- not return proxies when
        querying parameters.

        Args:

            attr (str): The name of the attribute being queried
            attr_val (Any): The value of the attribute
            parameter_proxy_cache (Dict[str, Any]): A cache of attr names to proxies

        Return:

            The return value from the getattr call.
        """
    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        """
        Create ``placeholder`` nodes corresponding to the signature of the ``root``
        Module. This method introspects root's signature and emits those
        nodes accordingly, also supporting ``*args`` and ``**kwargs``.
        """
    root: Incomplete
    graph: Incomplete
    tensor_attrs: dict[_ConstantAttributeType, str]
    def trace(self, root: torch.nn.Module | Callable[..., Any], concrete_args: dict[str, Any] | None = None) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
    def __deepcopy__(self, memo): ...
    def _proxy_placeholder(self, name, concrete_args, sig, fn_for_analysis): ...

_wrapped_fns_to_patch: dict[tuple[int, str], dict]
_wrapped_methods_to_patch: list[tuple[type, str]]

def _find_proxy(*objects_to_search):
    """
    Recursively search a data structure for a Proxy() and return it,
    return None if not found.
    """
def _create_wrapped_func(orig_fn): ...
def _create_wrapped_method(cls, name): ...

class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any
    new_fn: Any
    def revert(self) -> None: ...
    def patch(self) -> None: ...

class _PatchedFnSetItem(_PatchedFn):
    def revert(self) -> None: ...
    def patch(self) -> None: ...

class _PatchedFnDel(_PatchedFn):
    def revert(self) -> None: ...
    def patch(self) -> None: ...

class _PatchedFnSetAttr(_PatchedFn):
    def revert(self) -> None: ...
    def patch(self) -> None: ...

class _Patcher:
    patches_made: list[_PatchedFn]
    visited: set[int]
    def __init__(self) -> None: ...
    def patch(self, frame_dict: dict[str, Any], name: str, new_fn: Callable, deduplicate: bool = True):
        """
        Replace frame_dict[name] with new_fn until we exit the context manager.
        """
    def patch_method(self, cls: type, name: str, new_fn: Callable, deduplicate: bool = True):
        """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
    def visit_once(self, thing: Any):
        """Return True on the first call to with thing, otherwise false"""
    def revert_all_patches(self):
        """
        Remove all the stored patcheds. It doesn't modify patches_made.
        """
    def reapply_all_patches(self):
        """
        Patch all the stored patcheds. It doesn't modify patches_made.
        """
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None:
        """
        Undo all the changes made via self.patch() and self.patch_method()
        """

CURRENT_PATCHER: _Patcher | None

@contextlib.contextmanager
def _new_patcher() -> Generator[Incomplete]: ...
@contextlib.contextmanager
def _maybe_revert_all_patches() -> Generator[None]: ...
def _patch_wrapped_functions(patcher: _Patcher):
    """
    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap
    the listed global functions in the `_create_wrapped_func` wrapper.
    """
def _autowrap_check(patcher: _Patcher, frame_dict: dict[str, Any], function_ids: set[int]):
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
def wrap(fn_or_name: str | Callable):
    '''
    This function can be called at module-level scope to register fn_or_name as a "leaf function".
    A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being
    traced through::

        # foo/bar/baz.py
        def my_custom_function(x, y):
            return x * x + y * y


        torch.fx.wrap("my_custom_function")


        def fn_to_be_traced(x, y):
            # When symbolic tracing, the below call to my_custom_function will be inserted into
            # the graph rather than tracing it.
            return my_custom_function(x, y)

    This function can also equivalently be used as a decorator::

        # foo/bar/baz.py
        @torch.fx.wrap
        def my_custom_function(x, y):
            return x * x + y * y

    A wrapped function can be thought of a "leaf function", analogous to the concept of
    "leaf modules", that is, they are functions that are left as calls in the FX trace
    rather than traced through.

    Args:

        fn_or_name (Union[str, Callable]): The function or name of the global function to insert into the
            graph when it\'s called
    '''
def symbolic_trace(root: torch.nn.Module | Callable[..., Any], concrete_args: dict[str, Any] | None = None) -> GraphModule:
    '''
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    ``concrete_args`` allows you to partially specialize your function, whether it\'s to remove control flow or data structures.

    For example::

        def f(a, b):
            if b == True:
                return a
            else:
                return a * 2

    FX can typically not trace through this due to the presence of control
    flow. However, we can use `concrete_args` to specialize on the value of
    `b` to trace through this::

        f = fx.symbolic_trace(f, concrete_args={"b": False})
        assert f(3, False) == 6

    Note that although you can still pass in different values of `b`, they will be ignored.

    We can also use `concrete_args` to eliminate data-structure handling from
    our function. This will use pytrees to flatten your input. To avoid
    overspecializing, pass in `fx.PH` for values that shouldn\'t be
    specialized. For example::

        def f(x):
            out = 0
            for v in x.values():
                out += v
            return out


        f = fx.symbolic_trace(
            f, concrete_args={"x": {"a": fx.PH, "b": fx.PH, "c": fx.PH}}
        )
        assert f({"a": 1, "b": 2, "c": 4}) == 7


    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    '''
@wrap
def _assert_is_none(value, msg) -> None: ...
