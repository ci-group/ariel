import torch
from .graph import Graph
from torch._C import _NodeBase
from torch.fx.operator_schemas import ArgsKwargsPair
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

__all__ = ['Node', 'map_arg', 'map_aggregate', 'has_side_effect']

BaseArgumentTypes = str | int | float | bool | complex | torch.dtype | torch.Tensor | torch.device | torch.memory_format | torch.layout | torch._ops.OpOverload | torch.SymInt | torch.SymBool | torch.SymFloat
Target = Callable[..., Any] | str
ArgumentT = TypeVar('ArgumentT', bound=Argument)
_P = ParamSpec('_P')
_R = TypeVar('_R')

def has_side_effect(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...

class Node(_NodeBase):
    '''
    ``Node`` is the data structure that represents individual operations within
    a ``Graph``. For the most part, Nodes represent callsites to various entities,
    such as operators, methods, and Modules (some exceptions include nodes that
    specify function inputs and outputs). Each ``Node`` has a function specified
    by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

    - ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
      ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
      denoting the default parameter of the function input. ``kwargs`` is don\'t-care. Placeholders correspond to
      the function parameters (e.g. ``x``) in the graph printout.
    - ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
      fetch is assigned to. ``target`` is the fully-qualified name of the parameter\'s position in the module hierarchy.
      ``args`` and ``kwargs`` are don\'t-care
    - ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
      to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
      following the Python calling convention
    - ``call_module`` applies a module in the module hierarchy\'s ``forward()`` method to given arguments. ``name`` is
      as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
      ``args`` and ``kwargs`` represent the arguments to invoke the module on, *excluding the self argument*.
    - ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
      to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
      *including the self argument*
    - ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
      in the Graph printout.
    '''
    _args: tuple['Argument', ...]
    _kwargs: dict[str, 'Argument']
    graph: Graph
    name: str
    op: str
    target: Target
    _input_nodes: dict['Node', None]
    users: dict['Node', None]
    type: Any | None
    _sort_key: Any
    _repr_fn: Callable[[Node], str] | None
    meta: dict[str, Any]
    def __init__(self, graph: Graph, name: str, op: str, target: Target, args: tuple['Argument', ...], kwargs: dict[str, 'Argument'], return_type: Any | None = None) -> None:
        """
        Instantiate an instance of ``Node``. Note: most often, you want to use the
        Graph APIs, i.e. ``Graph.call_module``, ``Graph.call_method``, etc. rather
        than instantiating a ``Node`` directly.

        Args:
            graph (Graph): The ``Graph`` to which this ``Node`` should belong.

            name (str): The name to which the output of this ``Node`` should be assigned

            op (str): The opcode for this ``Node``. Can be one of 'placeholder',
                'call_method', 'call_module', 'call_function', 'get_attr',
                'output'

            target ('Target'): The target this op should call. See the broader
                ``Node`` docstring for more details.

            args (Tuple['Argument']): The args to be passed to ``target``

            kwargs (Dict[str, 'Argument']): The kwargs to be passed to ``target``

            return_type (Optional[Any]): The python type expression representing the
                type of the output of this node. This field can be used for
                annotation of values in the generated code or for other types
                of analyses.
        """
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    @property
    def next(self) -> Node:
        """
        Returns the next ``Node`` in the linked list of Nodes.

        Returns:

            The next ``Node`` in the linked list of Nodes.
        """
    @property
    def prev(self) -> Node:
        """
        Returns the previous ``Node`` in the linked list of Nodes.

        Returns:

            The previous ``Node`` in the linked list of Nodes.
        """
    def prepend(self, x: Node) -> None:
        """
        Insert x before this node in the list of nodes in the graph. Example::

            Before: p -> self
                    bx -> x -> ax
            After:  p -> x -> self
                    bx -> ax

        Args:
            x (Node): The node to put before this node. Must be a member of the same graph.
        """
    def __gt__(self, other: Node) -> bool: ...
    def __lt__(self, other: Node) -> bool: ...
    def __ge__(self, other: Node) -> bool: ...
    def __le__(self, other: Node) -> bool: ...
    def append(self, x: Node) -> None:
        """
        Insert ``x`` after this node in the list of nodes in the graph.
        Equivalent to ``self.next.prepend(x)``

        Args:
            x (Node): The node to put after this node. Must be a member of the same graph.
        """
    def _remove_from_list(self) -> None: ...
    @property
    def args(self) -> tuple[Argument, ...]:
        """
        The tuple of arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
    @args.setter
    def args(self, a: tuple[Argument, ...]) -> None:
        """
        Set the tuple of arguments to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
    @property
    def kwargs(self) -> dict[str, Argument]:
        """
        The dict of keyword arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
    @kwargs.setter
    def kwargs(self, k: dict[str, Argument]) -> None:
        """
        Set the dict of kwargs to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
    @property
    def all_input_nodes(self) -> list['Node']:
        """
        Return all Nodes that are inputs to this Node. This is equivalent to
        iterating over ``args`` and ``kwargs`` and only collecting the values that
        are Nodes.

        Returns:

            List of ``Nodes`` that appear in the ``args`` and ``kwargs`` of this
            ``Node``, in that order.
        """
    def update_arg(self, idx: int, arg: Argument) -> None:
        """
        Update an existing positional argument to contain the new value
        ``arg``. After calling, ``self.args[idx] == arg``.

        Args:

            idx (int): The index into ``self.args`` of the element to update
            arg (Argument): The new argument value to write into ``args``
        """
    def insert_arg(self, idx: int, arg: Argument) -> None:
        """
        Insert an positional argument to the argument list with given index.

        Args:

            idx (int): The index of the element in ``self.args`` to be inserted before.
            arg (Argument): The new argument value to insert into ``args``
        """
    def update_kwarg(self, key: str, arg: Argument) -> None:
        """
        Update an existing keyword argument to contain the new value
        ``arg``. After calling, ``self.kwargs[key] == arg``.

        Args:

            key (str): The key in ``self.kwargs`` of the element to update
            arg (Argument): The new argument value to write into ``kwargs``
        """
    @property
    def stack_trace(self) -> str | None:
        """
        Return the Python stack trace that was recorded during tracing, if any.
        When traced with fx.Tracer, this property is usually populated by
        `Tracer.create_proxy`. To record stack traces during tracing for debug purposes,
        set `record_stack_traces = True` on the `Tracer` instance.
        When traced with dynamo, this property will be populated by default by
        `OutputGraph.create_proxy`.

        stack_trace would have the innermost frame at the end of the string.
        """
    @stack_trace.setter
    def stack_trace(self, trace: str | None) -> None: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def _pretty_print_target(target: object) -> str:
        """
        Make target printouts more user-friendly.
        1) builtins will be printed as `builtins.xyz`
        2) operators will be printed as `operator.xyz`
        3) other callables will be printed with qualified name, e.g. torch.add
        """
    def format_node(self, placeholder_names: list[str] | None = None, maybe_return_typename: list[str] | None = None) -> str | None:
        """
        Return a descriptive string representation of ``self``.

        This method can be used with no arguments as a debugging
        utility.

        This function is also used internally in the ``__str__`` method
        of ``Graph``. Together, the strings in ``placeholder_names``
        and ``maybe_return_typename`` make up the signature of the
        autogenerated ``forward`` function in this Graph's surrounding
        GraphModule. ``placeholder_names`` and ``maybe_return_typename``
        should not be used otherwise.

        Args:
            placeholder_names: A list that will store formatted strings
                representing the placeholders in the generated
                ``forward`` function. Internal use only.
            maybe_return_typename: A single-element list that will store
                a formatted string representing the output of the
                generated ``forward`` function. Internal use only.

        Returns:
            str: If 1) we're using ``format_node`` as an internal helper
                in the ``__str__`` method of ``Graph``, and 2) ``self``
                is a placeholder Node, return ``None``. Otherwise,
                return a  descriptive string representation of the
                current Node.
        """
    def replace_all_uses_with(self, replace_with: Node, delete_user_cb: Callable[[Node], bool] = ..., *, propagate_meta: bool = False) -> list['Node']:
        """
        Replace all uses of ``self`` in the Graph with the Node ``replace_with``.

        Args:

            replace_with (Node): The node to replace all uses of ``self`` with.
            delete_user_cb (Callable): Callback that is called to determine
              whether a given user of the self node should be removed.
            propagate_meta (bool): Whether or not to copy all properties
              on the .meta field of the original node onto the replacement node.
              For safety, this is only valid to do if the replacement node
              doesn't already have an existing .meta field.

        Returns:

            The list of Nodes on which this change was made.
        """
    def is_impure(self, impure_random: bool = True) -> bool:
        """
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Args:
            impure_random (bool): Whether to treat rand op as impure.

        Returns:

            bool: If the op is impure or not.
        """
    def normalized_arguments(self, root: torch.nn.Module, arg_types: tuple[Any] | None = None, kwarg_types: dict[str, Any] | None = None, normalize_to_only_use_kwargs: bool = False) -> ArgsKwargsPair | None:
        """
        Returns normalized arguments to Python targets. This means that
        `args/kwargs` will be matched up to the module/functional's
        signature and return exclusively kwargs in positional order
        if `normalize_to_only_use_kwargs` is true.
        Also populates default values. Does not support positional-only
        parameters or varargs parameters.

        Supports module calls.

        May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

        Args:
            root (torch.nn.Module): Module upon which to resolve module targets.
            arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
            kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
            normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

        Returns:

            Returns NamedTuple ArgsKwargsPair, or `None` if not successful.
        """
    def replace_input_with(self, old_input: Node, new_input: Node) -> None:
        """
        Loop through input nodes of ``self``, and replace all instances of
        ``old_input`` with ``new_input``.

        Args:

            old_input (Node): The old input node to be replaced.
            new_input (Node): The new input node to replace ``old_input``.
        """
    def _rename(self, candidate: str) -> None: ...
    def __setattr__(self, name: str, value: Any) -> None: ...

def map_arg(a: ArgumentT, fn: Callable[[Node], Argument]) -> ArgumentT:
    """
    Apply fn recursively to each Node appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
def map_aggregate(a: ArgumentT, fn: Callable[[Argument], Argument]) -> ArgumentT:
    """
    Apply fn recursively to each object appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
