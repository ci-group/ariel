import torch
from .backend_config import BackendConfig as BackendConfig, get_tensorrt_backend_config as get_tensorrt_backend_config
from .fx.convert import convert as convert
from .fx.custom_config import ConvertCustomConfig as ConvertCustomConfig, FuseCustomConfig as FuseCustomConfig, PrepareCustomConfig as PrepareCustomConfig
from .fx.fuse import fuse as fuse
from .fx.graph_module import ObservedGraphModule as ObservedGraphModule
from .fx.prepare import prepare as prepare
from .fx.tracer import QuantizationTracer as QuantizationTracer, Scope as Scope, ScopeContextManager as ScopeContextManager
from .fx.utils import get_custom_module_class_keys as get_custom_module_class_keys, get_skipped_module_name_and_classes as get_skipped_module_name_and_classes
from .qconfig_mapping import QConfigMapping as QConfigMapping
from .utils import DEPRECATION_WARNING as DEPRECATION_WARNING
from torch.fx import GraphModule as GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY as _USER_PRESERVED_ATTRIBUTES_KEY
from typing import Any

def attach_preserved_attrs_to_model(model: GraphModule | torch.nn.Module, preserved_attrs: dict[str, Any]) -> None:
    """Store preserved attributes to the model.meta so that it can be preserved during deepcopy"""
def _check_is_graph_module(model: torch.nn.Module) -> None: ...
def _attach_meta_to_node_if_not_exist(model: GraphModule) -> None:
    """Attach meta field to all nodes of the graph if it does not exist,
    meta field is a field stores some meta information about the node, such
    as dtype and shape information for output of the node, this only exists
    if the program is captured by make_fx (used in quantize_pt2e flow), if
    the program is captured by torch.fx symbolic tracing, this field may not exist,
    so we add it here to avoid checking this all over the places
    """
def _swap_ff_with_fxff(model: torch.nn.Module) -> None:
    """Swap FloatFunctional with FXFloatFunctional"""
def _fuse_fx(model: GraphModule, is_qat: bool, fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None) -> GraphModule:
    """Internal helper function to fuse modules in preparation for quantization

    Args:
        model: GraphModule object from symbolic tracing (torch.fx.symbolic_trace)
    """
def _prepare_fx(model: torch.nn.Module, qconfig_mapping: QConfigMapping | dict[str, Any], is_qat: bool, example_inputs: tuple[Any, ...], prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None, _equalization_config: QConfigMapping | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None, is_standalone_module: bool = False) -> GraphModule:
    """Internal helper function for prepare_fx
        Args:
          `model`, `qconfig_mapping`, `prepare_custom_config`, `_equalization_config`:
          see docs for :func:`~torch.ao.quantization.prepare_fx`
          `is_standalone_module`: a boolean flag indicates whether we are
          quantizing a standalone module or not, a standalone module
          is a submodule of the parent module that is not inlined in the
    forward graph of the parent module,
          the way we quantize standalone module is described in:
          :func:`~torch.ao.quantization._prepare_standalone_module_fx`
    """
def _prepare_standalone_module_fx(model: torch.nn.Module, qconfig_mapping: QConfigMapping | dict[str, Any], is_qat: bool, example_inputs: tuple[Any, ...], prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None) -> GraphModule:
    """[Internal use only] Prepare a standalone module, so that it can be used when quantizing the
    parent module.
    standalone_module means it a submodule that is not inlined in parent module,
    and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module

    Returns:

        * model(GraphModule): prepared standalone module. It has these attributes in
          model.meta:

            * `standalone_module_input_quantized_idxs(List[Int])`: a list of
              indexes for the graph input that is expected to be quantized,
              same as input_quantized_idxs configuration provided
              for the standalone module
            * `standalone_module_output_quantized_idxs(List[Int])`: a list of
              indexs for the graph output that is quantized
              same as input_quantized_idxs configuration provided
              for the standalone module

    """
def fuse_fx(model: torch.nn.Module, fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None) -> GraphModule:
    """Fuse modules like conv+bn, conv+bn+relu etc, model must be in eval mode.
    Fusion rules are defined in torch.ao.quantization.fx.fusion_pattern.py

    Args:

        * `model` (torch.nn.Module): a torch.nn.Module model
        * `fuse_custom_config` (FuseCustomConfig): custom configurations for fuse_fx.
            See :class:`~torch.ao.quantization.fx.custom_config.FuseCustomConfig` for more details
    Example::

        from torch.ao.quantization import fuse_fx

        m = Model().eval()
        m = fuse_fx(m)

    """
def prepare_fx(model: torch.nn.Module, qconfig_mapping: QConfigMapping | dict[str, Any], example_inputs: tuple[Any, ...], prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None, _equalization_config: QConfigMapping | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None) -> GraphModule:
    ''' Prepare a model for post training quantization

    Args:
      * `model` (torch.nn.Module): torch.nn.Module model

      * `qconfig_mapping` (QConfigMapping): QConfigMapping object to configure how a model is
         quantized, see :class:`~torch.ao.quantization.qconfig_mapping.QConfigMapping`
         for more details

      * `example_inputs` (Tuple[Any, ...]): Example inputs for forward function of the model,
         Tuple of positional args (keyword args can be passed as positional args as well)

      * `prepare_custom_config` (PrepareCustomConfig): customization configuration for quantization tool.
          See :class:`~torch.ao.quantization.fx.custom_config.PrepareCustomConfig` for more details

      * `_equalization_config`: config for specifying how to perform equalization on the model

      * `backend_config` (BackendConfig): config that specifies how operators are quantized
         in a backend, this includes how the operators are observed,
         supported fusion patterns, how quantize/dequantize ops are
         inserted, supported dtypes etc. See :class:`~torch.ao.quantization.backend_config.BackendConfig` for more details

    Return:
      A GraphModule with observer (configured by qconfig_mapping), ready for calibration

    Example::

        import torch
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx

        class Submodule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Submodule()

            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x) + x
                return x

        # initialize a floating point model
        float_model = M().eval()

        # define calibration function
        def calibrate(model, data_loader):
            model.eval()
            with torch.no_grad():
                for image, target in data_loader:
                    model(image)

        # qconfig is the configuration for how we insert observers for a particular
        # operator
        # qconfig = get_default_qconfig("fbgemm")
        # Example of customizing qconfig:
        # qconfig = torch.ao.quantization.QConfig(
        #    activation=MinMaxObserver.with_args(dtype=torch.qint8),
        #    weight=MinMaxObserver.with_args(dtype=torch.qint8))
        # `activation` and `weight` are constructors of observer module

        # qconfig_mapping is a collection of quantization configurations, user can
        # set the qconfig for each operator (torch op calls, functional calls, module calls)
        # in the model through qconfig_mapping
        # the following call will get the qconfig_mapping that works best for models
        # that target "fbgemm" backend
        qconfig_mapping = get_default_qconfig_mapping("fbgemm")

        # We can customize qconfig_mapping in different ways.
        # e.g. set the global qconfig, which means we will use the same qconfig for
        # all operators in the model, this can be overwritten by other settings
        # qconfig_mapping = QConfigMapping().set_global(qconfig)
        # e.g. quantize the linear submodule with a specific qconfig
        # qconfig_mapping = QConfigMapping().set_module_name("linear", qconfig)
        # e.g. quantize all nn.Linear modules with a specific qconfig
        # qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)
        # for a more complete list, please see the docstring for :class:`torch.ao.quantization.QConfigMapping`
        # argument

        # example_inputs is a tuple of inputs, that is used to infer the type of the
        # outputs in the model
        # currently it\'s not used, but please make sure model(*example_inputs) runs
        example_inputs = (torch.randn(1, 3, 224, 224),)

        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        # `prepare_fx` inserts observers in the model based on qconfig_mapping and
        # backend_config. If the configuration for an operator in qconfig_mapping
        # is supported in the backend_config (meaning it\'s supported by the target
        # hardware), we\'ll insert observer modules according to the qconfig_mapping
        # otherwise the configuration in qconfig_mapping will be ignored
        #
        # Example:
        # in qconfig_mapping, user sets linear module to be quantized with quint8 for
        # activation and qint8 for weight:
        # qconfig = torch.ao.quantization.QConfig(
        #     observer=MinMaxObserver.with_args(dtype=torch.quint8),
        #     weight=MinMaxObserver.with-args(dtype=torch.qint8))
        # Note: current qconfig api does not support setting output observer, but
        # we may extend this to support these more fine grained control in the
        # future
        #
        # qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)
        # in backend config, linear module also supports in this configuration:
        # weighted_int8_dtype_config = DTypeConfig(
        #   input_dtype=torch.quint8,
        #   output_dtype=torch.quint8,
        #   weight_dtype=torch.qint8,
        #   bias_type=torch.float)

        # linear_pattern_config = BackendPatternConfig(torch.nn.Linear) \\\n        #    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \\\n        #    .add_dtype_config(weighted_int8_dtype_config) \\\n        #    ...

        # backend_config = BackendConfig().set_backend_pattern_config(linear_pattern_config)
        # `prepare_fx` will check that the setting requested by suer in qconfig_mapping
        # is supported by the backend_config and insert observers and fake quant modules
        # in the model
        prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)
        # Run calibration
        calibrate(prepared_model, sample_inference_data)
    '''
def prepare_qat_fx(model: torch.nn.Module, qconfig_mapping: QConfigMapping | dict[str, Any], example_inputs: tuple[Any, ...], prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None) -> GraphModule:
    '''Prepare a model for quantization aware training

    Args:
      * `model` (torch.nn.Module): torch.nn.Module model
      * `qconfig_mapping` (QConfigMapping): see :func:`~torch.ao.quantization.prepare_fx`
      * `example_inputs` (Tuple[Any, ...]): see :func:`~torch.ao.quantization.prepare_fx`
      * `prepare_custom_config` (PrepareCustomConfig): see :func:`~torch.ao.quantization.prepare_fx`
      * `backend_config` (BackendConfig): see :func:`~torch.ao.quantization.prepare_fx`

    Return:
      A GraphModule with fake quant modules (configured by qconfig_mapping and backend_config), ready for
      quantization aware training

    Example::

        import torch
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_qat_fx


        class Submodule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear(x)
                return x


        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Submodule()

            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x) + x
                return x


        # initialize a floating point model
        float_model = M().train()
        # (optional, but preferred) load the weights from pretrained model
        # float_model.load_weights(...)


        # define the training loop for quantization aware training
        def train_loop(model, train_data):
            model.train()
            for image, target in data_loader:
                ...


        # qconfig is the configuration for how we insert observers for a particular
        # operator
        # qconfig = get_default_qconfig("fbgemm")
        # Example of customizing qconfig:
        # qconfig = torch.ao.quantization.QConfig(
        #    activation=FakeQuantize.with_args(observer=MinMaxObserver.with_args(dtype=torch.qint8)),
        #    weight=FakeQuantize.with_args(observer=MinMaxObserver.with_args(dtype=torch.qint8)))
        # `activation` and `weight` are constructors of observer module

        # qconfig_mapping is a collection of quantization configurations, user can
        # set the qconfig for each operator (torch op calls, functional calls, module calls)
        # in the model through qconfig_mapping
        # the following call will get the qconfig_mapping that works best for models
        # that target "fbgemm" backend
        qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")

        # We can customize qconfig_mapping in different ways, please take a look at
        # the docstring for :func:`~torch.ao.quantization.prepare_fx` for different ways
        # to configure this

        # example_inputs is a tuple of inputs, that is used to infer the type of the
        # outputs in the model
        # currently it\'s not used, but please make sure model(*example_inputs) runs
        example_inputs = (torch.randn(1, 3, 224, 224),)

        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        # `prepare_qat_fx` inserts observers in the model based on qconfig_mapping and
        # backend_config, if the configuration for an operator in qconfig_mapping
        # is supported in the backend_config (meaning it\'s supported by the target
        # hardware), we\'ll insert fake_quantize modules according to the qconfig_mapping
        # otherwise the configuration in qconfig_mapping will be ignored
        # see :func:`~torch.ao.quantization.prepare_fx` for a detailed explanation of
        # how qconfig_mapping interacts with backend_config
        prepared_model = prepare_qat_fx(float_model, qconfig_mapping, example_inputs)
        # Run training
        train_loop(prepared_model, train_loop)

    '''
def _convert_fx(graph_module: GraphModule, is_reference: bool, convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None, is_standalone_module: bool = False, _remove_qconfig: bool = True, qconfig_mapping: QConfigMapping | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None, is_decomposed: bool = False, keep_original_weights: bool = False) -> GraphModule:
    """`is_standalone_module`: see docs in :func:`~torch.ao.quantization.prepare_standalone_module_fx`"""
def convert_fx(graph_module: GraphModule, convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None, _remove_qconfig: bool = True, qconfig_mapping: QConfigMapping | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None, keep_original_weights: bool = False) -> GraphModule:
    '''Convert a calibrated or trained model to a quantized model

    Args:
        * `graph_module` (torch.fx.GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :class:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig` for more details

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.

           The keys must include the ones in the qconfig_mapping passed to `prepare_fx` or `prepare_qat_fx`,
           with the same values or `None`. Additional keys can be specified with values set to `None`.

          For each entry whose value is set to None, we skip quantizing that entry in the model::

            qconfig_mapping = QConfigMapping
                .set_global(qconfig_from_prepare)
                .set_object_type(torch.nn.functional.add, None)  # skip quantizing torch.nn.functional.add
                .set_object_type(torch.nn.functional.linear, qconfig_from_prepare)
                .set_module_name("foo.bar", None)  # skip quantizing module "foo.bar"

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend, this includes quantization
            mode support (static/dynamic/weight_only), dtype support (quint8/qint8 etc.),
            observer placement for each operators and fused operators.
            See :class:`~torch.ao.quantization.backend_config.BackendConfig` for more details

    Return:
        A quantized model (torch.nn.Module)

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # convert_fx converts a calibrated/trained model to a quantized model for the
        # target hardware, this includes converting the model first to a reference
        # quantized model, and then lower the reference quantized model to a backend
        # Currently, the supported backends are fbgemm (onednn), qnnpack (xnnpack) and
        # they share the same set of quantized operators, so we are using the same
        # lowering procedure
        #
        # backend_config defines the corresponding reference quantized module for
        # the weighted modules in the model, e.g. nn.Linear
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        quantized_model = convert_fx(prepared_model)

    '''
def convert_to_reference_fx(graph_module: GraphModule, convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None, _remove_qconfig: bool = True, qconfig_mapping: QConfigMapping | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None) -> GraphModule:
    '''Convert a calibrated or trained model to a reference quantized model,
    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,
    reference quantized model is a standard representation of a quantized model provided
    by FX Graph Mode Quantization, it can be further lowered to run on the target
    hardware, like accelerators

    Args:
        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend. See
            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

    Return:
        A reference quantized model (GraphModule)

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        reference_quantized_model = convert_to_reference_fx(prepared_model)

    '''
def _convert_to_reference_decomposed_fx(graph_module: GraphModule, convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None, qconfig_mapping: QConfigMapping | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None) -> GraphModule:
    '''Convert a calibrated or trained model to a reference quantized model, with
    decomposed representation for quantized Tensor
    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,
    reference quantized model is a standard representation of a quantized model provided
    by FX Graph Mode Quantization, it can be further lowered to run on the target
    hardware, like accelerators

    Note: this is not public API

    Args:
        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend. See
            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

    Return:
        A reference quantized model (GraphModule) with operators working with decomposed quantized Tensor

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        reference_quantized_model = _convert_to_reference_decomposed_fx(prepared_model)

    '''
def _convert_standalone_module_fx(graph_module: GraphModule, is_reference: bool = False, convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None) -> GraphModule:
    """[Internal use only] Convert a model produced by :func:`~torch.ao.quantization.prepare_standalone_module_fx`
    and convert it to a quantized model

    Returns a quantized standalone module, whether input/output is quantized is
    specified by prepare_custom_config, with
    input_quantized_idxs, output_quantized_idxs, please
    see docs for prepare_fx for details
    """
