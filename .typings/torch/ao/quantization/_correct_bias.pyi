import torch.ao.ns._numeric_suite as ns
from _typeshed import Incomplete

__all__ = ['get_module', 'parent_child_names', 'get_param', 'MeanShadowLogger', 'bias_correction']

def get_module(model, name):
    """Given name of submodule, this function grabs the submodule from given model."""
def parent_child_names(name):
    """Split full name of submodule into parent submodule's full name and submodule's name."""
def get_param(module, attr):
    """Get the parameter given a module and attribute.

    Sometimes the weights/bias attribute gives you the raw tensor, but sometimes
    gives a function that will give you the raw tensor, this function takes care of that logic
    """

class MeanShadowLogger(ns.Logger):
    """Mean Logger for a Shadow module.

    A logger for a Shadow module whose purpose is to record the rolling mean
    of the data passed to the floating point and quantized models
    """
    count: int
    float_sum: Incomplete
    quant_sum: Incomplete
    def __init__(self) -> None:
        """Set up initial values for float and quantized stats, count, float sum, and quant sum."""
    def forward(self, x, y) -> None:
        """Compute the average of quantized and floating-point data from modules.

        The inputs x,y are output data from the quantized and floating-point modules.
        x is for the quantized module, y is for the floating point module
        """
    def clear(self) -> None: ...

def bias_correction(float_model, quantized_model, img_data, target_modules=..., neval_batches=None) -> None:
    """Perform bias correction on a module.

    Using numeric suite shadow module, the expected output of the floating point and quantized modules
    is recorded. Using that data the bias of supported modules is shifted to compensate for the drift caused
    by quantization
    Paper reference: https://arxiv.org/pdf/1906.04721.pdf (Section 4.2)

    Args:
        float_model: a trained model that serves as a reference to what bias correction should aim for
        quantized_model: quantized form of float_model that bias correction is to applied to
        img_data: calibration data to estimate the expected output (used to find quantization error)
        target_modules: specifies what submodules in quantized_model need bias correction (can be extended to
                unquantized submodules)
        neval_batches: a cap to the number of batches you want to be used for estimating the expected output
    """
