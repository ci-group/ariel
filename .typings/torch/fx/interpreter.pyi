import torch
from .graph import Graph
from .graph_module import GraphModule
from .node import Argument, Node, Target
from .proxy import Proxy
from _typeshed import Incomplete
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ['Interpreter', 'Transformer']

class Interpreter:
    '''
    An Interpreter executes an FX graph Node-by-Node. This pattern
    can be useful for many things, including writing code
    transformations as well as analysis passes.

    Methods in the Interpreter class can be overridden to customize
    the behavior of execution. The map of overrideable methods
    in terms of call hierarchy::

        run()
            +-- run_node
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass Interpreter like so::

            class NegSigmSwapInterpreter(Interpreter):
                def call_function(
                    self, target: Target, args: Tuple, kwargs: Dict
                ) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(target, args, kwargs)

                def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
                    if target == "neg":
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(target, args, kwargs)


            def fn(x):
                return torch.sigmoid(x).neg()


            gm = torch.fx.symbolic_trace(fn)
            input = torch.randn(3, 4)
            result = NegSigmSwapInterpreter(gm).run(input)
            torch.testing.assert_close(result, torch.neg(input).sigmoid())

    Args:
        module (torch.nn.Module): The module to be executed
        garbage_collect_values (bool): Whether to delete values after their last
            use within the Module\'s execution. This ensures optimal memory usage during
            execution. This can be disabled to, for example, examine all of the intermediate
            values in the execution by looking at the ``Interpreter.env`` attribute.
        graph (Optional[Graph]): If passed, the interpreter will execute this
            graph instead of `module.graph`, using the provided `module`
            argument to satisfy any requests for state.
    '''
    module: Incomplete
    submodules: Incomplete
    graph: Incomplete
    env: dict[Node, Any]
    name: str
    garbage_collect_values: Incomplete
    extra_traceback: bool
    user_to_last_uses: dict[Node, list[Node]]
    def __init__(self, module: torch.nn.Module, garbage_collect_values: bool = True, graph: Graph | None = None) -> None: ...
    args_iter: Iterator[Any]
    def run(self, *args, initial_env: dict[Node, Any] | None = None, enable_io_processing: bool = True) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
    def boxed_run(self, args_list):
        '''
        Run `module` via interpretation and return the result.  This uses the "boxed"
        calling convention, where you pass a list of arguments, which will be cleared
        by the interpreter.  This ensures that input tensors are promptly deallocated.
        '''
    @contextmanager
    def _set_current_node(self, node) -> Generator[None]: ...
    def run_node(self, n: Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
    def placeholder(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
    def get_attr(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The value of the attribute that was retrieved
        """
    def call_function(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
    def call_method(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
    def call_module(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        """
        Execute a ``call_module`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the module invocation
        """
    def output(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any:
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The return value referenced by the output node
        """
    def fetch_attr(self, target: str):
        """
        Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (str): The fully-qualified name of the attribute to fetch

        Return:
            Any: The value of the attribute.
        """
    def fetch_args_kwargs_from_env(self, n: Node) -> tuple[tuple, dict]:
        """
        Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``
        from the current execution environment.

        Args:
            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.

        Return:
            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.
        """
    def map_nodes_to_values(self, args: Argument, n: Node) -> Argument:
        """
        Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """

class Transformer(Interpreter):
    '''
    ``Transformer`` is a special type of interpreter that produces a
    new ``Module``. It exposes a ``transform()`` method that returns
    the transformed ``Module``. ``Transformer`` does not require
    arguments to run, as ``Interpreter`` does. ``Transformer`` works
    entirely symbolically.

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass ``Transformer`` like so::

            class NegSigmSwapXformer(Transformer):
                def call_function(
                    self,
                    target: "Target",
                    args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any],
                ) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(target, args, kwargs)

                def call_method(
                    self,
                    target: "Target",
                    args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any],
                ) -> Any:
                    if target == "neg":
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(target, args, kwargs)


            def fn(x):
                return torch.sigmoid(x).neg()


            gm = torch.fx.symbolic_trace(fn)

            transformed: torch.nn.Module = NegSigmSwapXformer(gm).transform()
            input = torch.randn(3, 4)
            torch.testing.assert_close(transformed(input), torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The ``Module`` to be transformed.
    '''
    new_graph: Incomplete
    graph: Incomplete
    tensor_attrs: dict[torch.Tensor, str]
    tracer: Incomplete
    def __init__(self, module) -> None: ...
    def placeholder(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Proxy:
        """
        Execute a ``placeholder`` node. In ``Transformer``, this is
        overridden to insert a new ``placeholder`` into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
    def get_attr(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Proxy:
        """
        Execute a ``get_attr`` node. In ``Transformer``, this is
        overridden to insert a new ``get_attr`` node into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
    def call_module(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any: ...
    def call_function(self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]) -> Any: ...
    def transform(self) -> GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
