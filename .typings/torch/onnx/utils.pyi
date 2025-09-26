import contextlib
import inspect
import torch
import torch._C._onnx as _C_onnx
import torch.jit
from collections.abc import Collection, Mapping, Sequence
from torch import _C
from typing import Any, Callable

__all__ = ['select_model_mode_for_export', 'disable_apex_o2_state_dict_hook', 'setup_onnx_logging', 'exporter_context', 'export', 'model_signature', 'warn_on_static_input_change', 'unpack_quantized_tensor', 'unconvertible_ops', 'register_custom_op_symbolic', 'unregister_custom_op_symbolic']

@contextlib.contextmanager
def select_model_mode_for_export(model, mode: _C_onnx.TrainingMode):
    """A context manager to temporarily set the training mode of ``model``
    to ``mode``, resetting it when we exit the with-block.

    .. deprecated:: 2.7
        Please set training mode before exporting the model.

    Args:
        model: Same type and meaning as ``model`` arg to :func:`export`.
        mode: Same type and meaning as ``training`` arg to :func:`export`.
    """
@contextlib.contextmanager
def disable_apex_o2_state_dict_hook(model: torch.nn.Module | torch.jit.ScriptFunction):
    """A context manager to temporarily disable the Apex O2 hook that returns.

    .. deprecated:: 2.7
        Please remove usage of this function.
    """
@contextlib.contextmanager
def setup_onnx_logging(verbose: bool):
    """A context manager to temporarily set the ONNX logging verbosity.

    .. deprecated:: 2.7
        Please remove usage of this function.
    """
@contextlib.contextmanager
def exporter_context(model, mode: _C_onnx.TrainingMode, verbose: bool):
    """A context manager to temporarily set the training mode of ``model``
    to ``mode``, disable the Apex O2 hook, and set the ONNX logging verbosity.

    .. deprecated:: 2.7
        Please set training mode before exporting the model.
    """
def export(model: torch.nn.Module | torch.jit.ScriptModule | torch.jit.ScriptFunction, args: tuple[Any, ...] | torch.Tensor, f: str, *, kwargs: dict[str, Any] | None = None, export_params: bool = True, verbose: bool = False, training: _C_onnx.TrainingMode = ..., input_names: Sequence[str] | None = None, output_names: Sequence[str] | None = None, operator_export_type: _C_onnx.OperatorExportTypes = ..., opset_version: int | None = None, do_constant_folding: bool = True, dynamic_axes: Mapping[str, Mapping[int, str]] | Mapping[str, Sequence[int]] | None = None, keep_initializers_as_inputs: bool | None = None, custom_opsets: Mapping[str, int] | None = None, export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = False, autograd_inlining: bool = True) -> None:
    '''Exports a model into ONNX format.

    If ``model`` is not a :class:`torch.jit.ScriptModule` nor a
    :class:`torch.jit.ScriptFunction`, this runs
    ``model`` once in order to convert it to a TorchScript graph to be exported
    (the equivalent of :func:`torch.jit.trace`). Thus this has the same limited support
    for dynamic control flow as :func:`torch.jit.trace`.

    Args:
        model: The model to be exported.
        args:

            args can be structured either as:

            1. ONLY A TUPLE OF ARGUMENTS::

                args = (x, y, z)

            The tuple should contain model inputs such that ``model(*args)`` is a valid
            invocation of the model. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.

            2. A TENSOR::

                args = torch.Tensor([1])

            This is equivalent to a 1-ary tuple of that Tensor.

            3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED ARGUMENTS::

                args = (x, {"y": input_y, "z": input_z})

            All but the last element of the tuple will be passed as non-keyword arguments,
            and named arguments will be set from the last element. If a named argument is
            not present in the dictionary, it is assigned the default value, or None if a
            default value is not provided.

            .. warning::
                This behavior will be deprecated in a future release. Please use the
                kwargs argument instead.

            .. note::
                If a dictionary is the last element of the args tuple, it will be
                interpreted as containing named arguments. In order to pass a dict as the
                last non-keyword arg, provide an empty dict as the last element of the args
                tuple. For example, instead of::

                    torch.onnx.export(
                        model,
                        (
                            x,
                            # WRONG: will be interpreted as named arguments
                            {y: z},
                        ),
                        "test.onnx.pb",
                    )

                Write::

                    torch.onnx.export(model, (x, {y: z}, {}), "test.onnx.pb")

        f: Path to the output ONNX model file. E.g. "model.onnx".
        kwargs: Named arguments to the model.
        export_params: If True, all parameters will
            be exported. Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, with the ordering as specified by ``model.state_dict().values()``
        verbose: if True, prints a description of the
            model being exported to stdout. In addition, the final ONNX graph will include the
            field ``doc_string``` from the exported model which mentions the source code locations
            for ``model``. If True, ONNX exporter logging will be turned on.
        training:
            * ``TrainingMode.EVAL``: export the model in inference mode.
            * ``TrainingMode.PRESERVE``: export the model in inference mode if model.training is
                False and in training mode if model.training is True.
            * ``TrainingMode.TRAINING``: export the model in training mode. Disables optimizations
                which might interfere with training.
        input_names (list of str, default empty list): names to assign to the
            input nodes of the graph, in order.
        output_names (list of str, default empty list): names to assign to the
            output nodes of the graph, in order.
        operator_export_type (enum, default OperatorExportTypes.ONNX):

            .. warning::
                This option will be deprecated in a future release. Future exported
                graphs will always use the default opset domain.

            * ``OperatorExportTypes.ONNX``: Export all ops as regular ONNX ops
                (in the default opset domain).
            * ``OperatorExportTypes.ONNX_FALLTHROUGH``: Try to convert all ops
                to standard ONNX ops in the default opset domain. If unable to do so
                (e.g. because support has not been added to convert a particular torch op to ONNX),
                fall back to exporting the op into a custom opset domain without conversion. Applies
                to `custom ops <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
                as well as ATen ops. For the exported model to be usable, the runtime must support
                these non-standard ops.
            * ``OperatorExportTypes.ONNX_ATEN``: All ATen ops (in the TorchScript namespace "aten")
                are exported as ATen ops (in opset domain "org.pytorch.aten").
                `ATen <https://pytorch.org/cppdocs/#aten>`_ is PyTorch\'s built-in tensor library, so
                this instructs the runtime to use PyTorch\'s implementation of these ops.

                .. warning::

                    Models exported this way are probably runnable only by Caffe2.

                    This may be useful if the numeric differences in implementations of operators are
                    causing large differences in behavior between PyTorch and Caffe2 (which is more
                    common on untrained models).

            * ``OperatorExportTypes.ONNX_ATEN_FALLBACK``: Try to export each ATen op
                (in the TorchScript namespace "aten") as a regular ONNX op. If we are unable to do so
                (e.g. because support has not been added to convert a particular torch op to ONNX),
                fall back to exporting an ATen op. See documentation on OperatorExportTypes.ONNX_ATEN for
                context.
                For example::

                    graph(%0 : Float):
                    %3 : int = prim::Constant[value=0]()
                    # conversion unsupported
                    %4 : Float = aten::triu(%0, %3)
                    # conversion supported
                    %5 : Float = aten::mul(%4, %0)
                    return (%5)

                Assuming ``aten::triu`` is not supported in ONNX, this will be exported as::

                    graph(%0 : Float):
                    %1 : Long() = onnx::Constant[value={0}]()
                    # not converted
                    %2 : Float = aten::ATen[operator="triu"](%0, %1)
                    # converted
                    %3 : Float = onnx::Mul(%2, %0)
                    return (%3)

                .. warning::

                    Models exported this way are probably runnable only by Caffe2.

        opset_version (int, default 18): The version of the
            `default (ai.onnx) opset <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
            to target. Must be >= 7.
        do_constant_folding: Apply the constant-folding optimization.
            Constant-folding will replace some of the ops that have all constant inputs
            with pre-computed constant nodes.
        dynamic_axes:

            By default the exported model will have the shapes of all input and output tensors
            set to exactly match those given in ``args``. To specify axes of tensors as
            dynamic (i.e. known only at run-time), set ``dynamic_axes`` to a dict with schema:

            * KEY (str): an input or output name. Each name must also be provided in ``input_names`` or
                ``output_names``.
            * VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
                list, each element is an axis index.

            For example::

                class SumModule(torch.nn.Module):
                    def forward(self, x):
                        return torch.sum(x, dim=1)


                torch.onnx.export(
                    SumModule(),
                    (torch.ones(2, 2),),
                    "onnx.pb",
                    input_names=["x"],
                    output_names=["sum"],
                )

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                ...

            While::

                torch.onnx.export(
                    SumModule(),
                    (torch.ones(2, 2),),
                    "onnx.pb",
                    input_names=["x"],
                    output_names=["sum"],
                    dynamic_axes={
                        # dict value: manually named axes
                        "x": {0: "my_custom_axis_name"},
                        # list value: automatic names
                        "sum": [0],
                    },
                )

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_param: "my_custom_axis_name"  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_param: "sum_dynamic_axes_1"  # axis 0
                ...

        keep_initializers_as_inputs: If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.
            This may allow for better optimizations (e.g. constant folding) by
            backends/runtimes.

            If True, `deduplicate_initializers` pass will not be executed. This means
            initializers with duplicated values will not be deduplicated and
            will be treated as distinct inputs to the graph. This allows different
            input initializers to be supplied at the runtime following export.

            If ``opset_version < 9``, initializers MUST be part of graph
            inputs and this argument will be ignored and the behavior will be
            equivalent to setting this argument to True.

        custom_opsets (dict[str, int], default empty dict): A dict with schema:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in this dictionary,
            the opset version is set to 1. Only custom opset domain name and version should be
            indicated through this argument.

        export_modules_as_functions: Flag to enable
            exporting all ``nn.Module`` forward calls as local functions in ONNX. Or a set to indicate the
            particular types of modules to export as local functions in ONNX.
            This feature requires ``opset_version`` >= 15, otherwise the export will fail. This is because
            ``opset_version`` < 15 implies IR version < 8, which means no local function support.
            Module variables will be exported as function attributes. There are two categories of function
            attributes.

            1. Annotated attributes: class variables that have type annotations via
            `PEP 526-style <https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations>`_
            will be exported as attributes.
            Annotated attributes are not used inside the subgraph of ONNX local function because
            they are not created by PyTorch JIT tracing, but they may be used by consumers
            to determine whether or not to replace the function with a particular fused kernel.

            2. Inferred attributes: variables that are used by operators inside the module. Attribute names
            will have prefix "inferred::". This is to differentiate from predefined attributes retrieved from
            python module annotations. Inferred attributes are used inside the subgraph of ONNX local function.

            * ``False`` (default): export ``nn.Module`` forward calls as fine grained nodes.
            * ``True``: export all ``nn.Module`` forward calls as local function nodes.
            * Set of type of nn.Module: export ``nn.Module`` forward calls as local function nodes,
                only if the type of the ``nn.Module`` is found in the set.

        autograd_inlining: Flag used to control whether to inline autograd functions.
            Refer to https://github.com/pytorch/pytorch/pull/74765 for more details.

    Raises:
        :class:`torch.onnx.errors.CheckerError`: If the ONNX checker detects an invalid ONNX graph.
        :class:`torch.onnx.errors.UnsupportedOperatorError`: If the ONNX graph cannot be exported because it
            uses an operator that is not supported by the exporter.
        :class:`torch.onnx.errors.OnnxExporterError`: Other errors that can occur during export.
            All errors are subclasses of :class:`errors.OnnxExporterError`.
    '''
def warn_on_static_input_change(input_states) -> None:
    """Warns that changes to input dictionaries and strings won't take effect in the traced ONNX graph.

    We accept dictionaries and strings as ONNX inputs, but they should be only for
    configuration use. we detect here if these inputs are modified, and if so we warn
    the user that the changes won't take effect in the traced ONNX graph.
    """
def unpack_quantized_tensor(value, cast_onnx_accepted: bool = True): ...
def unconvertible_ops(model, args, training: _C_onnx.TrainingMode = ..., opset_version: int | None = None) -> tuple[_C.Graph, list[str]]:
    '''Returns an approximated list of all ops that are yet supported by :mod:`torch.onnx`.

    .. deprecated:: 2.5
        Unconvertible ops are not definitive. Please remove usage of this function.

    The list is approximated because some ops may be removed during the conversion
    process and don\'t need to be converted. Some other ops may have partial support
    that will fail conversion with particular inputs. Please open a Github Issue
    for op support requests.

    Args:
        model: Same as the `model` parameter in :func:`torch.onnx.export`.
        args: Same as the `args` parameter in :func:`torch.onnx.export`.
        training: Same as the `training` parameter in :func:`torch.onnx.export`.
        opset_version: Same as the `opset_version` parameter in :func:`torch.onnx.export`.

    Returns:
        The JIT graph and a list of unconvertible ops in the format of "domain::op".
    '''
def register_custom_op_symbolic(symbolic_name: str, symbolic_fn: Callable, opset_version: int):
    '''Registers a symbolic function for a custom operator.

    When the user registers symbolic for custom/contrib ops,
    it is highly recommended to add shape inference for that operator via setType API,
    otherwise the exported graph may have incorrect shape inference in some extreme cases.
    An example of setType is `test_aten_embedding_2` in `test_operators.py`.

    See "Custom Operators" in the module documentation for an example usage.

    Args:
        symbolic_name (str): The name of the custom operator in "<domain>::<op>"
            format.
        symbolic_fn (Callable): A function that takes in the ONNX graph and
            the input arguments to the current operator, and returns new
            operator nodes to add to the graph.
        opset_version (int): The ONNX opset version in which to register.
    '''
def unregister_custom_op_symbolic(symbolic_name: str, opset_version: int):
    '''Unregisters ``symbolic_name``.

    See "Custom Operators" in the module documentation for an example usage.

    Args:
        symbolic_name (str): The name of the custom operator in "<domain>::<op>"
            format.
        opset_version (int): The ONNX opset version in which to unregister.
    '''
def model_signature(model: torch.nn.Module | Callable) -> inspect.Signature: ...
