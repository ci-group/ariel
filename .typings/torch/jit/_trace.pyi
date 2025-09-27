import contextlib
import torch
from _typeshed import Incomplete
from collections.abc import Generator
from enum import Enum
from torch._jit_internal import _get_model_id as _get_model_id, _qualified_name as _qualified_name, get_callable_argument_names as get_callable_argument_names, is_scripting as is_scripting
from torch.autograd import function as function
from torch.jit._script import ScriptModule as ScriptModule, _CachedForward as _CachedForward, script as script
from torch.jit._state import _enabled as _enabled, _python_cu as _python_cu
from torch.nn import Module as Module
from torch.testing._comparison import default_tolerances as default_tolerances
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten
R = TypeVar('R', covariant=True)
P = ParamSpec('P')

def _create_interpreter_name_lookup_fn(frames_up: int = 1): ...
def _unique_state_dict(module, keep_vars: bool = False): ...

class ONNXTracedModule(torch.nn.Module):
    inner: Incomplete
    strict: Incomplete
    _force_outplace: Incomplete
    _return_inputs: Incomplete
    _return_inputs_states: Incomplete
    def __init__(self, inner, strict: bool = True, force_outplace: bool = False, return_inputs: bool = False, return_inputs_states: bool = False) -> None: ...
    def forward(self, *args: torch.Tensor): ...

def _clone_inputs(args): ...

_JIT_TIME: Incomplete
_JIT_DISABLE: Incomplete
_JIT_STATS: Incomplete

@contextlib.contextmanager
def _time(trace_name, name, time: bool = True) -> Generator[None]: ...
def verify(model, args, loss_fn=..., devices=None):
    """
    Verify that a JIT compiled model has the same behavior as its uncompiled version along with its backwards pass.

    If your model returns multiple outputs,
    you must also specify a `loss_fn` to produce a loss for which
    the backwards will be computed.

    This function has side-effects (e.g., it executes your model / saves and loads
    parameters), so don't expect the model to come out exactly the same as what
    you passed in.

    Args:
        model (compiled torch.nn.Module or function): the module/function to be
            verified.  The module/function definition MUST have been decorated with
            `@torch.jit.compile`.
        args (tuple or Tensor): the positional arguments to pass to the
            compiled function/module to be verified.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        loss_fn (function, optional): the loss function to be applied to
            the output of the model, before backwards is invoked.  By default,
            we assume that a model returns a single result, and we :func:`torch.sum`
            before calling backwards; if this is inappropriate, you can pass your
            own loss function.  Note that if a model returns a tuple of results,
            these are passed as separate positional arguments to `loss_fn`.
        devices (iterable of device IDs, optional): the GPU devices which the
            compiled module will be run on.  This determines the RNG state we
            must save when running both compiled and uncompiled versions of the model.
    """
def _verify_equal(xs, ys) -> None: ...
def indent(s): ...

class TracingCheckError(Exception):
    message: str
    def __init__(self, graph_diff_error, tensor_compare_error, extra_msg=None) -> None: ...

def _check_trace(check_inputs, func, traced_func, check_tolerance, strict, force_outplace, is_trace_module, _module_class, example_inputs_is_kwarg: bool = False): ...

class TracerWarning(Warning):
    @staticmethod
    def ignore_lib_warnings() -> None: ...

def make_tuple(example_inputs): ...
def make_module(mod, _module_class, _compilation_unit): ...
def wrap_check_inputs(check_inputs): ...
def analyze_ts_result_with_export_result(export, trace): ...
def _trace_impl(func, example_inputs=None, optimize=None, check_trace: bool = True, check_inputs=None, check_tolerance: float = 1e-05, strict: bool = True, _force_outplace: bool = False, _module_class=None, _compilation_unit=..., example_kwarg_inputs=None, _store_inputs: bool = True): ...

class _ExportType(str, Enum):
    DIRECT_EXPORT = 'DIRECT_EXPORT'
    TRACE_AND_EXPORT = 'TRACE_AND_EXPORT'
    SOURCE_TO_SOURCE = 'SOURCE_TO_SOURCE'
    def __str__(self) -> str: ...

class _ExportOutcome(str, Enum):
    SUCCESS = 'SUCCESS'
    FAILED_TO_EXPORT = 'FAILED_TO_EXPORT'
    FAILED_TO_RUN = 'FAILED_TO_RUN'
    ACCURACY_ERROR = 'ACCURACY_ERROR'
    def __str__(self) -> str: ...

def trace(func, example_inputs=None, optimize=None, check_trace: bool = True, check_inputs=None, check_tolerance: float = 1e-05, strict: bool = True, _force_outplace: bool = False, _module_class=None, _compilation_unit=..., example_kwarg_inputs=None, _store_inputs: bool = True):
    """
    Trace a function and return an executable  or :class:`ScriptFunction` that will be optimized using just-in-time compilation.

    Tracing is ideal for code that operates only on
    ``Tensor``\\\\s and lists, dictionaries, and
    tuples of ``Tensor``\\\\s.

    Using `torch.jit.trace` and `torch.jit.trace_module`, you can turn an
    existing module or Python function into a TorchScript
    :class:`ScriptFunction` or :class:`ScriptModule`. You must provide example
    inputs, and we run the function, recording the operations performed on all
    the tensors.

    * The resulting recording of a standalone function produces `ScriptFunction`.
    * The resulting recording of `nn.Module.forward` or `nn.Module` produces
      `ScriptModule`.

    This module also contains any parameters that the original
    module had as well.

    Warning:
        Tracing only correctly records functions and modules which are not data
        dependent (e.g., do not have conditionals on data in tensors) and do not have
        any untracked external dependencies (e.g., perform input/output or
        access global variables). Tracing only records operations done when the given
        function is run on the given tensors. Therefore, the returned
        `ScriptModule` will always run the same traced graph on any input. This
        has some important implications when your module is expected to run
        different sets of operations, depending on the input and/or the module
        state. For example,

        * Tracing will not record any control-flow like if-statements or loops.
          When this control-flow is constant across your module, this is fine
          and it often inlines the control-flow decisions. But sometimes the
          control-flow is actually part of the model itself. For instance, a
          recurrent network is a loop over the (possibly dynamic) length of an
          input sequence.
        * In the returned :class:`ScriptModule`, operations that have different
          behaviors in ``training`` and ``eval`` modes will always behave as if
          it is in the mode it was in during tracing, no matter which mode the
          `ScriptModule` is in.

        In cases like these, tracing would not be appropriate and
        :func:`scripting <torch.jit.script>` is a better choice. If you trace
        such models, you may silently get incorrect results on subsequent
        invocations of the model. The tracer will try to emit warnings when
        doing something that may cause an incorrect trace to be produced.

    Args:
        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`
            that will be run with `example_inputs`. `func` arguments and return
            values  must be tensors or (possibly nested) tuples that contain
            tensors. When a module is passed `torch.jit.trace`, only the
            ``forward`` method is run and traced (see :func:`torch.jit.trace
            <torch.jit.trace_module>` for details).

    Keyword arguments:
        example_inputs (tuple or torch.Tensor or None, optional): A tuple of example
            inputs that will be passed to the function while tracing.
            Default: ``None``. Either this argument or ``example_kwarg_inputs``
            should be specified. The resulting trace can be run with inputs of
            different types and shapes assuming the traced operations support those
            types and shapes. `example_inputs` may also be a single Tensor in which
            case it is automatically wrapped in a tuple. When the value is None,
            ``example_kwarg_inputs`` should be specified.

        check_trace (``bool``, optional): Check if the same inputs run through
            traced code produce the same outputs. Default: ``True``. You might want
            to disable this if, for example, your network contains non-
            deterministic ops or if you are sure that the network is correct despite
            a checker failure.

        check_inputs (list of tuples, optional): A list of tuples of input
            arguments that should be used to check the trace against what is
            expected. Each tuple is equivalent to a set of input arguments that
            would be specified in ``example_inputs``. For best results, pass in
            a set of checking inputs representative of the space of shapes and
            types of inputs you expect the network to see.  If not specified,
            the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance
            to use in the checker procedure.  This can be used to relax the
            checker strictness in the event that results diverge numerically
            for a known reason, such as operator fusion.
        strict (``bool``, optional): run the tracer in a strict mode or not
            (default: ``True``). Only turn this off when you want the tracer to
            record your mutable container types (currently ``list``/``dict``)
            and you are sure that the container you are using in your
            problem is a ``constant`` structure and does not get used as
            control flow (if, for) conditions.
        example_kwarg_inputs (dict, optional): This parameter is a pack of keyword
            arguments of example inputs that will be passed to the function while
            tracing. Default: ``None``. Either this argument or ``example_inputs``
            should be specified. The dict will be unpacking by the arguments name
            of the traced function. If the keys of the dict don't not match with
            the traced function's arguments name, a runtime exception will be raised.

    Returns:
        If `func` is `nn.Module` or ``forward`` of `nn.Module`, `trace` returns
        a :class:`ScriptModule` object with a single ``forward`` method
        containing the traced code.  The returned `ScriptModule` will
        have the same set of sub-modules and parameters as the original
        ``nn.Module``.  If ``func`` is a standalone function, ``trace``
        returns `ScriptFunction`.

    Example (tracing a function):

    .. testcode::

        import torch

        def foo(x, y):
            return 2 * x + y

        # Run `foo` with the provided inputs and record the tensor operations
        traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

        # `traced_foo` can now be run with the TorchScript interpreter or saved
        # and loaded in a Python-free environment

    Example (tracing an existing module)::

        import torch
        import torch.nn as nn


        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)


        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

    """

_trace_module_map: dict[Any, Any] | None

def trace_module(mod, inputs, optimize=None, check_trace: bool = True, check_inputs=None, check_tolerance: float = 1e-05, strict: bool = True, _force_outplace: bool = False, _module_class=None, _compilation_unit=..., example_inputs_is_kwarg: bool = False, _store_inputs: bool = True):
    '''
    Trace a module and return an executable :class:`ScriptModule` that will be optimized using just-in-time compilation.

    When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only
    the ``forward`` method is run and traced. With ``trace_module``, you can specify a dictionary of
    method names to example inputs to trace (see the ``inputs``) argument below.

    See :func:`torch.jit.trace <torch.jit.trace>` for more information on tracing.

    Args:
        mod (torch.nn.Module):  A ``torch.nn.Module`` containing methods whose names are
                                specified in ``inputs``. The given methods will be compiled
                                as a part of a single `ScriptModule`.
        inputs (dict):  A dict containing sample inputs indexed by method names in ``mod``.
                                The inputs will be passed to methods whose names correspond to inputs\'
                                keys while tracing.
                                ``{ \'forward\' : example_forward_input, \'method2\': example_method2_input}``
    Keyword arguments:
        check_trace (``bool``, optional): Check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.
        example_inputs_is_kwarg (``bool``, optional): This parameter indicate whether the example inputs is a pack
                                           pack of keyword arguments. Default: ``False``.

    Returns:
        A :class:`ScriptModule` object with a single ``forward`` method containing the traced code.
        When ``func`` is a ``torch.nn.Module``, the returned :class:`ScriptModule` will have the same set of
        sub-modules and parameters as ``func``.

    Example (tracing a module with multiple methods)::

        import torch
        import torch.nn as nn


        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight


        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

        # Trace specific methods on a module (specified in `inputs`), constructs
        # a `ScriptModule` with `forward` and `weighted_kernel_sum` methods
        inputs = {
            "forward": example_forward_input,
            "weighted_kernel_sum": example_weight,
        }
        module = torch.jit.trace_module(n, inputs)

    '''
def is_tracing():
    """Return a boolean value.

    Returns ``True`` in tracing (if a function is called during the
    tracing of code with ``torch.jit.trace``) and ``False`` otherwise.
    """

class TracedModule(ScriptModule):
    _disable_script_meta: bool
    def __init__(self, orig, id_set=None, _compilation_unit=None) -> None: ...
    def forward(self, *args, **kwargs) -> None: ...
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value): ...
    def _get_name(self): ...
    def extra_repr(self): ...

class TopLevelTracedModule(TracedModule):
    forward: Callable[..., Any]
    def _reconstruct(self, cpp_module) -> None:
        """
        Re-construct an instance of TopLevelTracedModule using an instance of a C++ module.

        Args:
            cpp_module: The C++ module that this TopLevelTracedModule will be rebuilt around.
        """

def _script_if_tracing(fn: Callable[P, R]) -> Callable[P, R]: ...
def _get_trace_graph(f, args=(), kwargs=None, strict: bool = True, _force_outplace: bool = False, return_inputs: bool = False, _return_inputs_states: bool = False):
    """Return a tuple on tracing a function or model.

    .. warning::
        This function is internal-only and should only be used by the ONNX
        exporter. If you are trying to get a graph through tracing, please go
        through the public API instead::

            trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
            trace_graph = trace.graph

    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value. If return_inputs,
    also returns the trace inputs as part of the tuple

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Args:
        f (torch.nn.Module or function): the function or module
            to be traced.
        args (tuple or Tensor): the positional arguments to pass to the
            function/module to be traced.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        kwargs (dict): the keyword arguments to pass to the function/module
            to be traced.

    Example (trace a cell):

    .. testcode::

        trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
    """
