from torch.ao.quantization.observer import _is_activation_post_process

__all__ = ['get_default_custom_config_dict', 'propagate_qconfig_', 'add_quant_dequant', 'prepare', 'quantize', 'quantize_dynamic', 'prepare_qat', 'quantize_qat', 'convert', 'swap_module']

is_activation_post_process = _is_activation_post_process

def get_default_custom_config_dict():
    """Defines the default custom config dict."""
def propagate_qconfig_(module, qconfig_dict=None, prepare_custom_config_dict=None) -> None:
    """Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)
        prepare_custom_config_dict: dictionary for custom handling of modules
            see docs for :func:`~torch.ao.quantization.prepare_fx`

    Return:
        None, module is modified inplace with qconfig attached
    """
def add_quant_dequant(module):
    """Wrap the leaf child module in QuantWrapper if it has a valid qconfig
    Note that this function will modify the children of module inplace and it
    can return a new module which wraps the input module as well.

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize

    Return:
        Either the inplace modified module with submodules wrapped in
        `QuantWrapper` based on qconfig or a new `QuantWrapper` module which
        wraps the input module, the latter case only happens when the input
        module is a leaf module and we want to quantize it.
    """
def prepare(model, inplace: bool = False, allow_list=None, observer_non_leaf_module_list=None, prepare_custom_config_dict=None):
    '''Prepares a copy of the model for quantization calibration or quantization-aware training.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    The model will be attached with observer or fake quant modules, and qconfig
    will be propagated.

    Args:
        `model`: input model to be modified in-place
        `inplace`: carry out model transformations in-place, the original module is mutated
        `allow_list`: list of quantizable modules
        `observer_non_leaf_module_list`: list of non-leaf modules we want to add observer
        `prepare_custom_config_dict`: customization configuration dictionary for prepare function

    .. code-block:: python

       # Example of prepare_custom_config_dict:
       prepare_custom_config_dict = {
           # user will manually define the corresponding observed
           # module class which has a from_float class method that converts
           # float custom module to observed custom module
           "float_to_observed_custom_module_class": {CustomModule: ObservedCustomModule}
       }

    '''
def quantize(model, run_fn, run_args, mapping=None, inplace: bool = False):
    """Quantize the input float model with post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        model: input float model
        run_fn: a calibration function for calibrating the prepared model
        run_args: positional arguments for `run_fn`
        inplace: carry out model transformations in-place, the original module is mutated
        mapping: correspondence between original module types and quantized counterparts

    Return:
        Quantized model.
    """
def quantize_dynamic(model, qconfig_spec=None, dtype=..., mapping=None, inplace: bool = False):
    """Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization
    by default is performed for layers with large weights size - i.e. Linear and RNN variants.

    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        model: input model
        qconfig_spec: Either:

            - A dictionary that maps from name or type of submodule to quantization
              configuration, qconfig applies to all submodules of a given
              module unless qconfig for the submodules are specified (when the
              submodule already has qconfig attribute). Entries in the dictionary
              need to be QConfig instances.

            - A set of types and/or submodule names to apply dynamic quantization to,
              in which case the `dtype` argument is used to specify the bit-width

        inplace: carry out model transformations in-place, the original module is mutated
        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced

    """
def prepare_qat(model, mapping=None, inplace: bool = False):
    """
    Prepares a copy of the model for quantization calibration or
    quantization-aware training and converts it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
def quantize_qat(model, run_fn, run_args, inplace: bool = False):
    """Do quantization aware training and output a quantized model

    Args:
        model: input model
        run_fn: a function for evaluating the prepared model, can be a
                function that simply runs the prepared model or a training
                loop
        run_args: positional arguments for `run_fn`

    Return:
        Quantized model.
    """
def convert(module, mapping=None, inplace: bool = False, remove_qconfig: bool = True, is_reference: bool = False, convert_custom_config_dict=None, use_precomputed_fake_quant: bool = False):
    '''Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class. And remove qconfig at the
    end if remove_qconfig is set to True.

    Args:
        `module`: prepared and calibrated module
        `mapping`: a dictionary that maps from source module type to target
                   module type, can be overwritten to allow swapping user defined
                   Modules
        `inplace`: carry out model transformations in-place, the original module
                   is mutated
        `convert_custom_config_dict`: custom configuration dictionary for convert function
        `use_precomputed_fake_quant`: a flag to enable use of precomputed fake quant

    .. code-block:: python

       # Example of convert_custom_config_dict:
       convert_custom_config_dict = {
           # user will manually define the corresponding quantized
           # module class which has a from_observed class method that converts
           # observed custom module to quantized custom module
           "observed_to_quantized_custom_module_class": {
               ObservedCustomModule: QuantizedCustomModule
           }
       }

    '''
def swap_module(mod, mapping, custom_module_class_mapping, use_precomputed_fake_quant: bool = False):
    """Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
