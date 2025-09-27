import torch
import torch.utils._pytree as pytree
import types
from ._symbolic_trace import Tracer
from .graph_module import GraphModule
from .node import Argument, Node, Target
from _typeshed import Incomplete
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple

__all__ = ['PythonCode', 'CodeGen', 'Graph']

TransformCodeFunc = Callable[[list[str]], list[str]]

class _CustomBuiltin(NamedTuple):
    """Additional objs that we add to every graph's globals.

    The repr() for some standard library objects is not valid Python code without
    an import. For common objects of this sort, we bundle them in the globals of
    every FX graph.
    """
    import_str: str
    obj: Any

class _Namespace:
    """A context for associating names uniquely with objects.

    The following invariants are enforced:
    - Each object gets a single name.
    - Each name is unique within a given namespace.
    - Names generated do not shadow builtins, unless the object is indeed that builtin.
    """
    _obj_to_name: dict[Any, str]
    _used_names: set[str]
    _base_count: dict[str, int]
    def __init__(self) -> None: ...
    def create_name(self, candidate: str, obj: Any | None) -> str:
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
    def associate_name_with_obj(self, name: str, obj: Any):
        """Associate a unique name with an object.

        Neither `name` nor `obj` should be associated already.
        """
    def _rename_object(self, obj: Any, name: str): ...

@dataclass
class PythonCode:
    """
    Represents all the information necessary to exec or save a graph as Python code.
    """
    src: str
    globals: dict[str, Any]
    _lineno_map: dict[int, int | None] | None

class _InsertPoint:
    graph: Incomplete
    def __init__(self, graph, new_insert) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, tb: types.TracebackType | None) -> None: ...

class _node_list:
    graph: Incomplete
    direction: Incomplete
    def __init__(self, graph: Graph, direction: Literal['_prev', '_next'] = '_next') -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __reversed__(self): ...

class _PyTreeInfo(NamedTuple):
    """
    Contains extra info stored when we're using Pytrees
    """
    orig_args: list[str]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec | None

@dataclass(frozen=True)
class _ParsedStackTrace:
    """
    Represents the top-most frame of a parsed stack trace
    """
    file: str
    lineno: str
    name: str
    code: str
    def get_summary_str(self): ...

class CodeGen:
    _sym_repr: Callable[[torch.types.PySymType], str]
    _body_transformer: TransformCodeFunc | None
    _func_name: str
    def __init__(self) -> None: ...
    def gen_fn_def(self, free_vars: list[str], maybe_return_annotation: str) -> str:
        """
        Given the free variables and a return annotation, generates the beginning of the FX function.
        By default, `gen_fn_def(['a', 'b'], '') == 'def {self._func_name}(a, b):'`
        """
    def generate_output(self, output_args: Argument) -> str:
        """
        Given the output arguments, generates the return statement of the FX function.
        Note: The returned statement should not be indented.
        """
    def process_inputs(self, *args: Any) -> Any:
        """
        Transforms the inputs so that the graph can take them as arguments, as
        non-default codegen may result in the inputs to the function being
        different from the inputs to the graph.

        If the graph was directly runnable, this invariant should hold true
        `f.graph.process_outputs(f.graph(*f.graph.process_inputs(*inputs))) == f(*inputs)`
        """
    def process_outputs(self, outputs: Any) -> Any:
        """
        Transforms the outputs of the graph to be identical to the codegen.

        See ``process_inputs`` for more details.
        """
    def additional_globals(self) -> list[tuple[str, Any]]:
        """
        If your codegen uses extra global values, add tuples of (identifier,reference to the value) here.
        For example, return ['List', typing.List] if you need ``List`` in the global context.
        """
    def _gen_python_code(self, nodes, root_module: str, namespace: _Namespace, *, verbose: bool = False, include_stride: bool = False, include_device: bool = False, colored: bool = False) -> PythonCode: ...

class _PyTreeCodeGen(CodeGen):
    pytree_info: _PyTreeInfo
    def __init__(self, pytree_info: _PyTreeInfo) -> None: ...
    def process_inputs(self, *inputs: Any) -> Any: ...
    def process_outputs(self, out: Any) -> Any: ...
    def gen_fn_def(self, free_vars, maybe_return_annotation): ...
    def generate_output(self, output_args): ...

class _FindNodesLookupTable:
    """
    Side table for the graph for the purpose of doing fast queries
    """
    table: dict[tuple[str, Target | None], dict[Node, None]]
    def __init__(self) -> None: ...
    def _key(self, node) -> tuple[str, Target | None]: ...
    def __contains__(self, node) -> bool: ...
    def insert(self, node: Node) -> None: ...
    def remove(self, node: Node) -> None: ...
    def find_nodes(self, *, op: str, target: Target | None = None): ...

class Graph:
    """
    ``Graph`` is the main data structure used in the FX Intermediate Representation.
    It consists of a series of ``Node`` s, each representing callsites (or other
    syntactic constructs). The list of ``Node`` s, taken together, constitute a
    valid Python function.

    For example, the following code

    .. code-block:: python

        import torch
        import torch.fx


        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return torch.topk(
                    torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3
                )


        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

    Will produce the following Graph::

        print(gm.graph)

    .. code-block:: text

        graph(x):
            %linear_weight : [num_users=1] = self.linear.weight
            %add_1 : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
            %linear_1 : [num_users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
            %relu_1 : [num_users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
            %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
            %topk_1 : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
            return topk_1

    For the semantics of operations represented in the ``Graph``, please see :class:`Node`.
    """
    _root: Node
    _used_names: dict[str, int]
    _insert: Incomplete
    _len: int
    _graph_namespace: Incomplete
    _owning_module: Incomplete
    _tracer_cls: Incomplete
    _tracer_extras: Incomplete
    _codegen: Incomplete
    _co_fields: dict[str, Any]
    _find_nodes_lookup_table: Incomplete
    def __init__(self, owning_module: GraphModule | None = None, tracer_cls: type['Tracer'] | None = None, tracer_extras: dict[str, Any] | None = None) -> None:
        """
        Construct an empty Graph.
        """
    @property
    def owning_module(self): ...
    @owning_module.setter
    def owning_module(self, mod: GraphModule | None): ...
    @property
    def nodes(self) -> _node_list:
        """
        Get the list of Nodes that constitute this Graph.

        Note that this ``Node`` list representation is a doubly-linked list. Mutations
        during iteration (e.g. delete a Node, add a Node) are safe.

        Returns:

            A doubly-linked list of Nodes. Note that ``reversed`` can be called on
            this list to switch iteration order.
        """
    def output_node(self) -> Node: ...
    def find_nodes(self, *, op: str, target: Target | None = None, sort: bool = True):
        """
        Allows for fast query of nodes

        Args:

            op (str): the name of the operation

            target (Optional[Target]): the target of the node. For call_function,
                the target is required. For other ops, the target is optional.

            sort (bool): whether to return nodes in the order they appear on
                         on the graph.

        Returns:

            Iteratable of nodes with the requested op and target.
        """
    def graph_copy(self, g: Graph, val_map: dict[Node, Node], return_output_node: bool = False) -> Argument | None:
        """
        Copy all nodes from a given graph into ``self``.

        Args:

            g (Graph): The source graph from which to copy Nodes.

            val_map (Dict[Node, Node]): a dictionary that will be populated with a mapping
                from nodes in ``g`` to nodes in ``self``. Note that ``val_map`` can be passed
                in with values in it already to override copying of certain values.

        Returns:

            The value in ``self`` that is now equivalent to the output value in ``g``,
            if ``g`` had an ``output`` node. ``None`` otherwise.
        """
    def __deepcopy__(self, memo=None) -> Graph:
        """
        Explicitly implement __deepcopy__ to prevent excessive recursion depth
        from the default implementation. This uses graph_copy to copy the nodes
        in an iterative way, rather than recursive. It also populates the
        memoization table to prevent unnecessary copies (e.g. references to
        nodes or other parts of the Graph from a custom GraphModule implementation.
        """
    def create_node(self, op: str, target: Target, args: tuple['Argument', ...] | None = None, kwargs: dict[str, 'Argument'] | None = None, name: str | None = None, type_expr: Any | None = None) -> Node:
        """
        Create a ``Node`` and add it to the ``Graph`` at the current insert-point.
        Note that the current insert-point can be set via :meth:`Graph.inserting_before`
        and :meth:`Graph.inserting_after`.

        Args:
            op (str): the opcode for this Node. One of 'call_function', 'call_method', 'get_attr',
                'call_module', 'placeholder', or 'output'. The semantics of these opcodes are
                described in the ``Graph`` docstring.

            args (Optional[Tuple[Argument, ...]]): is a tuple of arguments to this node.

            kwargs (Optional[Dict[str, Argument]]): the kwargs of this Node

            name (Optional[str]): an optional string name for the ``Node``.
                This will influence the name of the value assigned to in the
                Python generated code.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted node.
        """
    def process_inputs(self, *args):
        """
        Processes args so that they can be passed to the FX graph.
        """
    def process_outputs(self, out): ...
    def erase_node(self, to_erase: Node) -> None:
        """
        Erases a ``Node`` from the ``Graph``. Throws an exception if
        there are still users of that node in the ``Graph``.

        Args:

            to_erase (Node): The ``Node`` to erase from the ``Graph``.
        """
    def inserting_before(self, n: Node | None = None):
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits::

            with g.inserting_before(n):
                ...  # inserting before node n
            ...  # insert point restored to what it was previously
            g.inserting_before(n)  #  set the insert point permanently

        Args:

            n (Optional[Node]): The node before which to insert. If None this will insert before
                the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
    def inserting_after(self, n: Node | None = None):
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits::

            with g.inserting_after(n):
                ...  # inserting after node n
            ...  # insert point restored to what it was previously
            g.inserting_after(n)  #  set the insert point permanently

        Args:

            n (Optional[Node]): The node before which to insert. If None this will insert after
                the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
    def placeholder(self, name: str, type_expr: Any | None = None, default_value: Any = ...) -> Node:
        """
        Insert a ``placeholder`` node into the Graph. A ``placeholder`` represents
        a function input.

        Args:

            name (str): A name for the input value. This corresponds to the name
                of the positional argument to the function this ``Graph`` represents.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have. This is needed in some
                cases for proper code generation (e.g. when the function is used
                subsequently in TorchScript compilation).

            default_value (Any): The default value this function argument should take
                on. NOTE: to allow for `None` as a default value, `inspect.Signature.empty`
                should be passed as this argument to specify that the parameter does _not_
                have a default value.

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
    def get_attr(self, qualified_name: str, type_expr: Any | None = None) -> Node:
        """
        Insert a ``get_attr`` node into the Graph. A ``get_attr`` ``Node`` represents the
        fetch of an attribute from the ``Module`` hierarchy.

        Args:

            qualified_name (str): the fully-qualified name of the attribute to be retrieved.
                For example, if the traced Module has a submodule named ``foo``, which has a
                submodule named ``bar``, which has an attribute named ``baz``, the qualified
                name ``foo.bar.baz`` should be passed as ``qualified_name``.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.


        Returns:

            The newly-created and inserted ``get_attr`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
    def call_module(self, module_name: str, args: tuple['Argument', ...] | None = None, kwargs: dict[str, 'Argument'] | None = None, type_expr: Any | None = None) -> Node:
        """
        Insert a ``call_module`` ``Node`` into the ``Graph``. A ``call_module`` node
        represents a call to the forward() function of a ``Module`` in the ``Module``
        hierarchy.

        Args:

            module_name (str): The qualified name of the ``Module`` in the ``Module``
                hierarchy to be called. For example, if the traced ``Module`` has a
                submodule named ``foo``, which has a submodule named ``bar``, the
                qualified name ``foo.bar`` should be passed as ``module_name`` to
                call that module.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this should *not* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted ``call_module`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
    def call_method(self, method_name: str, args: tuple['Argument', ...] | None = None, kwargs: dict[str, 'Argument'] | None = None, type_expr: Any | None = None) -> Node:
        """
        Insert a ``call_method`` ``Node`` into the ``Graph``. A ``call_method`` node
        represents a call to a given method on the 0th element of ``args``.

        Args:

            method_name (str): The name of the method to apply to the self argument.
                For example, if args[0] is a ``Node`` representing a ``Tensor``,
                then to call ``relu()`` on that ``Tensor``, pass ``relu`` to ``method_name``.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this *should* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly created and inserted ``call_method`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
    def call_function(self, the_function: Callable[..., Any], args: tuple['Argument', ...] | None = None, kwargs: dict[str, 'Argument'] | None = None, type_expr: Any | None = None, name: str | None = None) -> Node:
        """
        Insert a ``call_function`` ``Node`` into the ``Graph``. A ``call_function`` node
        represents a call to a Python callable, specified by ``the_function``.

        Args:

            the_function (Callable[..., Any]): The function to be called. Can be any PyTorch
                operator, Python function, or member of the ``builtins`` or ``operator``
                namespaces.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called function.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called function

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

            name (Optional[str]): The name of the node. If not specified, set to None

        Returns:

            The newly created and inserted ``call_function`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
    def node_copy(self, node: Node, arg_transform: Callable[[Node], 'Argument'] = ...) -> Node:
        """
        Copy a node from one graph into another. ``arg_transform`` needs to transform arguments from
        the graph of node to the graph of self. Example::

            # Copying all the nodes in `g` into `new_graph`
            g: torch.fx.Graph = ...
            new_graph = torch.fx.graph()
            value_remap = {}
            for node in g.nodes:
                value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

        Args:

            node (Node): The node to copy into ``self``.

            arg_transform (Callable[[Node], Argument]): A function that transforms
                ``Node`` arguments in node's ``args`` and ``kwargs`` into the
                equivalent argument in ``self``. In the simplest case, this should
                retrieve a value out of a table mapping Nodes in the original
                graph to ``self``.
        """
    def output(self, result: Argument, type_expr: Any | None = None):
        """
        Insert an ``output`` ``Node`` into the ``Graph``. An ``output`` node represents
        a ``return`` statement in Python code. ``result`` is the value that should
        be returned.

        Args:

            result (Argument): The value to be returned.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        .. note::

            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
    def _target_to_str(self, target: Target) -> str: ...
    def python_code(self, root_module: str, *, verbose: bool = False, include_stride: bool = False, include_device: bool = False, colored: bool = False) -> PythonCode:
        """
        Turn this ``Graph`` into valid Python code.

        Args:

            root_module (str): The name of the root module on which to look-up
                qualified name targets. This is usually 'self'.

        Returns:

            A PythonCode object, consisting of two fields:
                src: the Python source code representing the object
                globals: a dictionary of global names in `src` -> the objects that they reference.
        """
    def _python_code(self, root_module: str, namespace: _Namespace, *, verbose: bool = False, include_stride: bool = False, include_device: bool = False, colored: bool = False) -> PythonCode: ...
    def __str__(self) -> str:
        """
        Return a human-readable (not machine-readable) string representation
        of this Graph
        """
    def print_tabular(self) -> None:
        """
        Prints the intermediate representation of the graph in tabular
        format. Note that this API requires the ``tabulate`` module to be
        installed.
        """
    def lint(self) -> None:
        """
        Runs various checks on this Graph to make sure it is well-formed. In
        particular:
        - Checks Nodes have correct ownership (owned by this graph)
        - Checks Nodes appear in topological order
        - If this Graph has an owning GraphModule, checks that targets
        exist in that GraphModule
        """
    def eliminate_dead_code(self, is_impure_node: Callable[[Node], bool] | None = None) -> bool:
        """
        Remove all dead code from the graph, based on each node's number of
        users, and whether the nodes have any side effects. The graph must be
        topologically sorted before calling.

        Args:
            is_impure_node (Optional[Callable[[Node], bool]]): A function that returns
            whether a node is impure. If this is None, then the default behavior is to
            use Node.is_impure.

        Returns:
          bool: Whether the graph was changed as a result of the pass.

        Example:

        Before dead code is eliminated, `a` from `a = x + 1` below has no users
        and thus can be eliminated from the graph without having an effect.

        .. code-block:: python

            def forward(self, x):
                a = x + 1
                return x + self.attr_1

        After dead code is eliminated, `a = x + 1` has been removed, and the rest
        of `forward` remains.

        .. code-block:: python

            def forward(self, x):
                return x + self.attr_1

        .. warning::

            Dead code elimination has some heuristics to avoid removing
            side-effectful nodes (see Node.is_impure) but in general coverage
            is very bad, so you should assume that this method is not sound
            to call unless you know that your FX graph consists entirely
            of functional operations or you supply your own custom
            function for detecting side-effectful nodes.
        """
    def set_codegen(self, codegen: CodeGen): ...
    def on_generate_code(self, make_transformer: Callable[[TransformCodeFunc | None], TransformCodeFunc]):
        '''Register a transformer function when python code is generated

        Args:
            make_transformer (Callable[[Optional[TransformCodeFunc]], TransformCodeFunc]):
                a function that returns a code transformer to be registered.
                This function is called by `on_generate_code` to obtain the
                code transformer.

                This function is also given as its input the currently
                registered code transformer (or None if nothing is registered),
                in case it is not desirable to overwrite it. This is useful to
                chain code transformers together.

        Returns:
            a context manager that when used in a `with` statement, to automatically
            restore the previously registered code transformer.

        Example:

        .. code-block:: python


            gm: fx.GraphModule = ...


            # This is a code transformer we want to register. This code
            # transformer prepends a pdb import and trace statement at the very
            # beginning of the generated torch.fx code to allow for manual
            # debugging with the PDB library.
            def insert_pdb(body):
                return ["import pdb; pdb.set_trace()\\n", *body]


            # Registers `insert_pdb`, and overwrites the current registered
            # code transformer (given by `_` to the lambda):
            gm.graph.on_generate_code(lambda _: insert_pdb)

            # Or alternatively, registers a code transformer which first
            # runs `body` through existing registered transformer, then
            # through `insert_pdb`:
            gm.graph.on_generate_code(
                lambda current_trans: (
                    lambda body: insert_pdb(
                        current_trans(body) if current_trans else body
                    )
                )
            )

            gm.recompile()
            gm(*inputs)  # drops into pdb


        This function can also be used as a context manager, with the benefit to
        automatically restores the previously registered code transformer:

        .. code-block:: python

            # ... continue from previous example

            with gm.graph.on_generate_code(lambda _: insert_pdb):
                # do more stuff with `gm`...
                gm.recompile()
                gm(*inputs)  # drops into pdb

            # now previous code transformer is restored (but `gm`\'s code with pdb
            # remains - that means you can run `gm` with pdb here too, until you
            # run next `recompile()`).
        '''
