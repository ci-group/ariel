from ._compatibility import compatibility as compatibility
from collections.abc import Generator
from contextlib import contextmanager
from torch.fx.graph_module import GraphModule as GraphModule, _format_import_block as _format_import_block, reduce_graph_module as reduce_graph_module, reduce_package_graph_module as reduce_package_graph_module
from torch.package import PackageExporter as PackageExporter, sys_importer as sys_importer

_use_lazy_graph_module_flag: bool
_force_skip_lazy_graph_module_flag: bool

@contextmanager
def _force_skip_lazy_graph_module() -> Generator[None]:
    """
    Skip using lazy graph module disregarding the setting of _use_lazy_graph_module.
    Use to skip _LazyGraphModule when testing inductor torchscript related backend.

    torch.jit.script a _LazyGraphModule results in following error:
        https://gist.github.com/shunting314/5143654c8084aed84ecd19b818258a69
    """
@contextmanager
def _use_lazy_graph_module(should_use: bool): ...
def _get_graph_module_cls(): ...
def _make_graph_module(*args, graph_module_cls=None, **kwargs): ...

class _LazyGraphModule(GraphModule):
    """
    The main difference between _LazyGraphModule and GraphModule is how recompile happens.
    GraphModule will do a 'recompile' call to generate python code and the forward method when it's
    constructed. Later on if the graph get updated, recompile method can be called again to refresh
    the saved python code and forward method.

    However in some cases especially in inductor, the recompilation can be a waste since we never
    check the python code for the graph module or call its forward method. A few more concreate
    examples regarding pattern matching fx passes in inductor:
    1. some passes will update the graph to be compiled and then call recompile on the GraphModule.
    2. some passes will trace small pattern function to search it in the graph being compiled and
       replace the match with the traced graph of a replacement function. The pattern graph and
       replacement graph are quite small but there are large amount of them. Doing GraphModule.recompile
       for them in GraphModule.__init__ is also a waste of time.

    However simply skip calling GraphModule.recompile in these scenarios is also dangeruous.
    People may want to check the python code or call the GraphModule's forward method for debugging purposes.

    The way _LazyGraphModule solves it is, we override the recompile method to just mark the
    need for recompilation but does not do the actual recompilation. Later on if people really
    access the compiled python code or call the GraphModule's forward method, we do the real
    recompilation.
    """
    @classmethod
    def from_graphmodule(cls, gm: GraphModule): ...
    @staticmethod
    def force_recompile(gm) -> None:
        """
        Sometimes we need force a recompile as a workaround
        - we want to do the real recompilation before symbolic_trace to avoid error:
            https://gist.github.com/shunting314/75549c2e82ae07ac1139c94a3583d259
        """
    def real_recompile(self) -> None: ...
    @classmethod
    def _needs_recompile(cls): ...
    def _lazy_forward(self, *args, **kwargs): ...
    forward = _lazy_forward
    def __reduce_package__(self, exporter: PackageExporter):
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
    def __reduce__(self):
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
    def _real_recompile(self): ...
    @classmethod
    def recompile(cls) -> None: ...
    @property
    def code(self) -> str: ...
    def __str__(self) -> str:
        """
        str(GraphModule) will access the _code attribute. Make sure recompile
        happens so _code attribute is available.
        """
