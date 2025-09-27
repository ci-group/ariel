import contextlib
import os
import torch
import traceback
from .graph import Graph, PythonCode
from _typeshed import Incomplete
from collections.abc import Generator
from torch.package import Importer, PackageExporter, PackageImporter
from typing import Any, Callable

__all__ = ['reduce_graph_module', 'reduce_package_graph_module', 'reduce_deploy_graph_module', 'GraphModule']

class _EvalCacheLoader:
    eval_cache: Incomplete
    next_id: int
    def __init__(self) -> None: ...
    def cache(self, src: str, globals: dict[str, Any], co_fields=None):
        """Store the source in a private cache, and add a lazy entry in linecache
        that allows the source to be retrieved by 'filename'.

        Args:
            src (str): The module source to cache
            globals (dict): The module globals

        Returns:
            str: The cache key (and dummy filename) generated for src.
        """
    def get_source(self, module_name) -> str | None: ...
    def _get_key(self): ...

def reduce_graph_module(body: dict[Any, Any], import_block: str) -> torch.nn.Module: ...
def reduce_package_graph_module(importer: PackageImporter, body: dict[Any, Any], generated_module_name: str) -> torch.nn.Module: ...
def reduce_deploy_graph_module(importer: PackageImporter, body: dict[Any, Any], import_block: str) -> torch.nn.Module: ...

class _CodeOnlyModule(torch.nn.Module):
    __dict__: Incomplete
    def __init__(self, body) -> None: ...

class _WrappedCall:
    cls: Incomplete
    cls_call: Incomplete
    def __init__(self, cls, cls_call) -> None: ...
    @staticmethod
    def _generate_error_message(frame_summary: traceback.FrameSummary) -> str: ...
    def __call__(self, obj, *args, **kwargs): ...

class GraphModule(torch.nn.Module):
    """
    GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a
    ``graph`` attribute, as well as ``code`` and ``forward`` attributes generated
    from that ``graph``.

    .. warning::

        When ``graph`` is reassigned, ``code`` and ``forward`` will be automatically
        regenerated. However, if you edit the contents of the ``graph`` without reassigning
        the ``graph`` attribute itself, you must call ``recompile()`` to update the generated
        code.
    """
    def __new__(cls, *args, **kwargs): ...
    training: Incomplete
    _tracer_cls: Incomplete
    _tracer_extras: Incomplete
    meta: dict[str, Any]
    _replace_hooks: list[Callable]
    _create_node_hooks: list[Callable]
    _erase_node_hooks: list[Callable]
    _deepcopy_hooks: list[Callable]
    def __init__(self, root: torch.nn.Module | dict[str, Any], graph: Graph, class_name: str = 'GraphModule') -> None:
        """
        Construct a GraphModule.

        Args:

            root (Union[torch.nn.Module, Dict[str, Any]):
                ``root`` can either be an nn.Module instance or a Dict mapping strings to any attribute type.
                In the case that ``root`` is a Module, any references to Module-based objects (via qualified
                name) in the Graph's Nodes' ``target`` field will be copied over from the respective place
                within ``root``'s Module hierarchy into the GraphModule's module hierarchy.
                In the case that ``root`` is a dict, the qualified name found in a Node's ``target`` will be
                looked up directly in the dict's keys. The object mapped to by the Dict will be copied
                over into the appropriate place within the GraphModule's module hierarchy.

            graph (Graph): ``graph`` contains the nodes this GraphModule should use for code generation

            class_name (str): ``name`` denotes the name of this GraphModule for debugging purposes. If it's unset, all
                error messages will report as originating from ``GraphModule``. It may be helpful to set this
                to ``root``'s original name or a name that makes sense within the context of your transform.
        """
    __jit_unused_properties__: Incomplete
    @property
    def graph(self) -> Graph:
        """
        Return the ``Graph`` underlying this ``GraphModule``
        """
    _graph: Incomplete
    @graph.setter
    def graph(self, g: Graph) -> None:
        """
        Set the underlying ``Graph`` for this ``GraphModule``. This will internally
        recompile the ``GraphModule`` so that the generated ``forward()`` function
        corresponds to ``g``
        """
    def to_folder(self, folder: str | os.PathLike, module_name: str = 'FxModule'):
        """Dumps out module to ``folder`` with ``module_name`` so that it can be
        imported with ``from <folder> import <module_name>``

        Args:

            folder (Union[str, os.PathLike]): The folder to write the code out to

            module_name (str): Top-level name to use for the ``Module`` while
                writing out the code
        """
    def add_submodule(self, target: str, m: torch.nn.Module) -> bool:
        """
        Adds the given submodule to ``self``.

        This installs empty Modules where none exist yet if they are
        subpaths of ``target``.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)
            m: The submodule itself; the actual object we want to
                install in the current Module

        Return:
            bool: Whether or not the submodule could be inserted. For
                this method to return True, each object in the chain
                denoted by ``target`` must either a) not exist yet,
                or b) reference an ``nn.Module`` (not a parameter or
                other attribute)
        """
    def delete_submodule(self, target: str) -> bool:
        """
        Deletes the given submodule from ``self``.

        The module will not be deleted if ``target`` is not a valid
        target.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)

        Returns:
            bool: Whether or not the target string referenced a
                submodule we want to delete. A return value of ``False``
                means that the ``target`` was not a valid reference to
                a submodule.
        """
    def delete_all_unused_submodules(self) -> None:
        '''
        Deletes all unused submodules from ``self``.

        A Module is considered "used" if any one of the following is
        true:
        1. It has children that are used
        2. Its forward is called directly via a ``call_module`` node
        3. It has a non-Module attribute that is used from a
        ``get_attr`` node

        This method can be called to clean up an ``nn.Module`` without
        manually calling ``delete_submodule`` on each unused submodule.
        '''
    @property
    def code(self) -> str:
        """
        Return the Python code generated from the ``Graph`` underlying this
        ``GraphModule``.
        """
    _in_spec: Incomplete
    _out_spec: Incomplete
    _code: Incomplete
    _lineno_map: Incomplete
    def recompile(self) -> PythonCode:
        """
        Recompile this GraphModule from its ``graph`` attribute. This should be
        called after editing the contained ``graph``, otherwise the generated
        code of this ``GraphModule`` will be out of date.
        """
    def __reduce_deploy__(self, importer: Importer): ...
    def __reduce_package__(self, exporter: PackageExporter): ...
    def __reduce__(self):
        """
        Serialization of GraphModule. We serialize only the generated code, not
        the underlying ``Graph``. This is because ``Graph`` does not have on-disk
        backward-compatibility guarantees, whereas Python source code does.
        On the deserialization side, we symbolically trace through the generated
        code to regenerate the underlying ``Graph``
        """
    def _deepcopy_init(self): ...
    def __deepcopy__(self, memo): ...
    def __copy__(self): ...
    def print_readable(self, print_output: bool = True, include_stride: bool = False, include_device: bool = False, colored: bool = False, *, fast_sympy_print: bool = False):
        """
        Return the Python code generated for current GraphModule and its children GraphModules
        """
    def __str__(self) -> str: ...
    def _replicate_for_data_parallel(self): ...
    @contextlib.contextmanager
    def _set_replace_hook(self, f) -> Generator[None]:
        """
        Takes a callable which will be called everytime when we replace a node
        to a new node, or change the node's name. Callable takes three arguments:
        the old node we're changing, and NAME of the new node, followed by the
        user node which consumes the old node to be replaced.
        """
    def _register_replace_node_hook(self, f) -> None:
        """
        Takes a callable which will be called everytime when we replace a node
        to a new node, or change the node's name. Callable takes three arguments:
        the old node we're changing, and NAME of the new node, followed by the
        user node which consumes the old node to be replaced.
        """
    def _unregister_replace_node_hook(self, f) -> None:
        """
        Takes a callable which was previously registered to be called everytime when we replace a node.
        This function will unregister that callable so it is no longer invoked on node replacement.
        """
    def _register_create_node_hook(self, f) -> None:
        """
        Takes a callable which will be called after we create a new node. The
        callable takes the newly created node as input and returns None.
        """
    def _unregister_create_node_hook(self, f) -> None:
        """
        Takes a callable which was previously registered to be called after we create a node.
        This function will unregister that callable so it is no longer invoked on node creation.
        """
    def _register_erase_node_hook(self, f) -> None:
        """
        Takes a callable which will be called after we erase a node. The
        callable takes the node that is being erased as input and returns None.
        """
    def _unregister_erase_node_hook(self, f) -> None:
        """
        Takes a callable which was previously registered to be called after we erase a node.
        This function will unregister that callable so it is no longer invoked on node erasure.
        """
    def _register_deepcopy_hook(self, f) -> None:
        """
        Takes a callable which will be called when we deepcopy this graph module. The
        callable takes the resulting deepcopied graph module.
        """
    def _unregister_deepcopy_hook(self, f) -> None:
        """
        Takes a callable which was previously registered to be called after deepcopy.
        This function will unregister that callable so it is no longer invoked on deepcopy.
        """
