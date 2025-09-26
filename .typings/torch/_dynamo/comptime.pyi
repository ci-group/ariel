import torch
from .exc import unimplemented_v2 as unimplemented_v2
from .variables import CellVariable as CellVariable
from .variables.constant import ConstantVariable as ConstantVariable
from .variables.tensor import SymNodeVariable as SymNodeVariable
from _typeshed import Incomplete
from torch.fx.experimental.symbolic_shapes import free_symbols as free_symbols

class ComptimeVar:
    """
    A ComptimeVar represents a Python value, at some particular point
    in time, in the Python code we are symbolically evaluating with
    torchdynamo.  This must be distinguished from a runtime value, as
    at compile-time there are some properties of the variable we
    do not know (for example, if the ComptimeVar represents a Tensor,
    we only know metadata about the tensor; we do NOT know what the
    actual data in the Tensor is.)
    """
    __variable: Incomplete
    def __init__(self, v) -> None: ...
    def as_proxy(self):
        """
        Returns an fx.Proxy (or tuple/list of fx.Proxy) representing
        this variable in the FX graph we are assembling to pass
        to the user compiler.

        This method only works for variables we actually track in
        the FX graph, aka Tensors (and ints, if you are compiling
        with dynamic shapes).  In particular, if you have a list
        or tuple of tensors, you will get a list/tuple of proxies
        (not a single proxy representing the entire list/tuple).
        """
    def is_proxy(self):
        """
        Returns True if as_proxy() would succeed.
        """
    def as_fake(self):
        '''
        Returns a "fake" value (either a FakeTensor or a SymInt)
        representing the variable in question.  This only works
        for variables that denote Tensor or int.  You can use
        this to query metadata; e.g., v.as_fake().size(0) will
        tell you the compile-time known size of the tensor.

        WARNING: Do NOT mutate the returned tensor.
        '''
    def size(self, dim: int | None = None) -> int | torch.SymInt:
        """
        Returns the size of the tensor (if dim is None) or the size
        at the dimension dim.  The returned size may be a SymInt.
        """
    def python_type(self):
        """
        Returns what type(v) would have returned for the variable
        at compile time.
        """
    def as_python_constant(self):
        """
        Returns the Python value this variable would have, but only if it is
        completely known at compile-time (e.g., it is constant).

        WARNING: Do NOT mutate the returned constant.  The returned constant
        may or may not correspond to the actual value this variable may take
        on at runtime; for example, if the variable in question is a constant
        list, we may return a copy of that list.
        """
    def is_python_constant(self):
        """
        Returns True if as_python_constant would succeed.
        """
    def is_dynamic(self): ...
    def force_static(self) -> None:
        """
        Forces that a value is static, inducing a guard on its specific value
        """
    def _i_will_not_complain_if_bc_breaks_VariableTracker(self):
        """
        Returns the internal data structure VariableTracker that Dynamo uses
        to represent variables at compile time.  There are no BC guarantees on
        this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if you rely on
        it.
        """
    def __repr__(self) -> str: ...

class ComptimeContext:
    """
    This context class provides access to a public API for Dynamo's internals.
    If there is something here you would find useful that is missing, please
    file a feature request at https://github.com/pytorch/pytorch/
    """
    __tx: Incomplete
    def __init__(self, tx) -> None: ...
    def get_local(self, name: str, *, stacklevel: int = 0) -> ComptimeVar:
        """
        Retrieve the compile-time known information about a local.
        """
    def graph_break(self, msg: str = 'ComptimeContext.graph_break') -> None:
        """
        Manually trigger a graph break
        """
    def graph(self):
        """
        Retrieve the partially constructed FX graph that would be
        passed to the user compiler after compilation.
        """
    def assert_static(self, val) -> None:
        """
        Asserts that the int is static (and not dynamic, per dynamic shapes)
        """
    def print_graph(self, *, verbose: bool = True, file=None) -> None:
        """
        Print the partially constructed FX graph that would be passed
        to the user compiler after compilation.
        """
    def parent(self): ...
    def __get_tx(self, stacklevel): ...
    def print(self, val, *, file=None) -> None: ...
    def print_disas(self, *, file=None, stacklevel: int = 0) -> None:
        """
        Print the current series of opcodes being executed (not including
        parent frames), including where you are in the particular opcode
        stream.
        """
    def print_value_stack(self, *, file=None, stacklevel: int = 0) -> None:
        """
        Print the current Python value stack.  Note that this is NOT the same
        as the traceback; use print_bt() to print that.  Note that at
        stacklevel=0, this will typically be empty, as comptime cannot
        currently be used in an expression context where there would be
        intermediates on the stack.  If you would find this useful, please
        file a bug at https://github.com/pytorch/pytorch/

        NB: Stack grows downwards in our print
        """
    def print_locals(self, *, file=None, stacklevel: int = 0) -> None:
        """
        Print all of the locals available in the current context.
        By default this view is very limited; you can get more information
        about any individual local using get_local().
        """
    def print_bt(self, *, file=None, stacklevel: int = 0) -> None:
        """
        Print the user code backtrace, starting at the beginning of the
        frame Dynamo started evaluating.  Note that this MAY NOT go all
        the way to the torch.compile invocation, as we may have done
        a graph break and are compiling an intermediate frame as the
        starting point.  If you think the other behavior would be better,
        file a bug at https://github.com/pytorch/pytorch/
        """
    def print_guards(self, *, file=None) -> None:
        """
        Print the currently installed guards for the Dynamo context.
        This does NOT include guards associated with variables that
        may or may not be installed in the future if those variables
        are used.
        """
    def _i_will_not_complain_if_bc_breaks_InstructionTranslator(self):
        """
        Returns the internal data structure InstructionTranslator that Dynamo
        uses to track state of symbolic evaluation.  There are no BC
        guarantees on this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if
        you rely on it.
        """
    def sleep(self, sec) -> None: ...

class _Comptime:
    @staticmethod
    def __call__(fn, fallback_fn=...) -> None:
        """fn gets called at compile time in TorchDynamo, calls fallback_fn otherwise"""
    @staticmethod
    def graph_break(): ...
    @staticmethod
    def print(e): ...
    @staticmethod
    def print_graph(): ...
    @staticmethod
    def print_disas(*, stacklevel: int = 0): ...
    @staticmethod
    def print_value_stack(*, stacklevel: int = 0): ...
    @staticmethod
    def print_value_stack_and_return(e, *, stacklevel: int = 0): ...
    @staticmethod
    def print_locals(*, stacklevel: int = 0): ...
    @staticmethod
    def print_bt(*, stacklevel: int = 0): ...
    @staticmethod
    def print_guards(): ...
    @staticmethod
    def assert_static(val): ...
    @staticmethod
    def force_static(val): ...
    @staticmethod
    def breakpoint() -> None:
        '''
        Like pdb breakpoint(), but drop into pdb whenever this line
        of code is compiled by dynamo.  Use it by putting
        this in your model code::

            from torch._dynamo.comptime import comptime

            comptime.breakpoint()

        And then, inside pdb, you can access \'ctx\' to query things
        about the compilation context::

            (Pdb) !ctx.print_bt()
            (Pdb) !ctx.print_locals()
            (Pdb) p ctx.get_local("attention").as_fake()
        '''
    @staticmethod
    def sleep(sec): ...

comptime: Incomplete
