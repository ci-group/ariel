__all__ = ['script_qconfig', 'script_qconfig_dict', 'fuse_conv_bn_jit', 'prepare_jit', 'prepare_dynamic_jit', 'convert_jit', 'convert_dynamic_jit', 'quantize_jit', 'quantize_dynamic_jit']

def script_qconfig(qconfig):
    """Instantiate the activation and weight observer modules and script
    them, these observer module instances will be deepcopied during
    prepare_jit step.
    """
def script_qconfig_dict(qconfig_dict):
    """Helper function used by `prepare_jit`.
    Apply `script_qconfig` for all entries in `qconfig_dict` that is
    not None.
    """
def fuse_conv_bn_jit(model, inplace: bool = False):
    """Fuse conv - bn module
    Works for eval model only.

    Args:
        model: TorchScript model from scripting or tracing
    """
def prepare_jit(model, qconfig_dict, inplace: bool = False): ...
def prepare_dynamic_jit(model, qconfig_dict, inplace: bool = False): ...
def convert_jit(model, inplace: bool = False, debug: bool = False, preserved_attrs=None): ...
def convert_dynamic_jit(model, inplace: bool = False, debug: bool = False, preserved_attrs=None): ...
def quantize_jit(model, qconfig_dict, run_fn, run_args, inplace: bool = False, debug: bool = False):
    '''Quantize the input float TorchScript model with
    post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, empty key means the qconfig will be applied
        to whole model unless it\'s overwritten by more specific configurations, the
        qconfig for each module is either found in the dictionary or fallback to
         the qconfig of parent module.

        Right now qconfig_dict is the only way to configure how the model is quantized,
        and it is done in the granularity of module, that is, we only support one type
        of qconfig for each torch.nn.Module, and the qconfig for sub module will
        override the qconfig for parent module, empty string means global configuration.
        `run_fn`: a calibration function for calibrating the prepared model
        `run_args`: positional arguments for `run_fn`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchSciprt model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization import quantize_jit

    ts_model = torch.jit.script(
        float_model.eval()
    )  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig("fbgemm")


    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)


    quantized_model = quantize_jit(
        ts_model, {"": qconfig}, calibrate, [data_loader_test]
    )
    ```
    '''
def quantize_dynamic_jit(model, qconfig_dict, inplace: bool = False, debug: bool = False):
    '''Quantize the input float TorchScript model with
    post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        descriptions in :func:`~torch.ao.quantization.quantize_jit`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchSciprt model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization import quantize_dynamic_jit

    ts_model = torch.jit.script(
        float_model.eval()
    )  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig("fbgemm")


    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)


    quantized_model = quantize_dynamic_jit(
        ts_model, {"": qconfig}, calibrate, [data_loader_test]
    )
    ```
    '''
