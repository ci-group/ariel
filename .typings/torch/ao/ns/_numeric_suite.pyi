import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch.ao.quantization import prepare as prepare
from torch.ao.quantization.quantization_mappings import get_default_compare_output_module_list as get_default_compare_output_module_list
from typing import Any, Callable

NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST: Incomplete

def _find_match(str_list: dict[str, Any] | list[str], key_str: str, postfix: str) -> str | None: ...
def compare_weights(float_dict: dict[str, Any], quantized_dict: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    '''Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys \'float\' and \'quantized\', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage::

        wt_compare_dict = compare_weights(float_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(
                key,
                compute_error(
                    wt_compare_dict[key]["float"],
                    wt_compare_dict[key]["quantized"].dequantize(),
                ),
            )

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys \'float\' and \'quantized\', containing the float and
        quantized weights
    '''
def _get_logger_dict_helper(mod: nn.Module, target_dict: dict[str, Any], prefix: str = '') -> None:
    """This is the helper function for get_logger_dict

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module
        target_dict: the dictionary used to save all logger stats
    """
def get_logger_dict(mod: nn.Module, prefix: str = '') -> dict[str, dict]:
    """Traverse the modules and save all logger stats into target dict.
    This is mainly used for quantization accuracy debug.

    Type of loggers supported:
        ShadowLogger: used to log the outputs of the quantized module and its matching float shadow module,
        OutputLogger: used to log the outputs of the modules

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module

    Return:
        target_dict: the dictionary used to save all logger stats

    """

class Logger(nn.Module):
    """Base class for stats logging"""
    stats: Incomplete
    dtype: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x) -> None:
        """
        """

class ShadowLogger(Logger):
    """Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """
    def __init__(self) -> None: ...
    def forward(self, x, y) -> None:
        """
        """

class OutputLogger(Logger):
    """Class used to log the outputs of the module"""
    def __init__(self) -> None: ...
    def forward(self, x):
        """
        """

def _convert_tuple_to_list(t: Any) -> Any: ...
def _dequantize_tensor_list(t: Any) -> Any: ...

class Shadow(nn.Module):
    """Shadow module attaches the float module to its matching quantized module
    as the shadow. Then it uses Logger module to process the outputs of both
    modules.

    Args:
        q_module: module quantized from float_module that we want to shadow
        float_module: float module used to shadow q_module
        logger_cls: type of logger used to process the outputs of q_module and
            float_module. ShadowLogger or custom loggers can be used.
    """
    orig_module: Incomplete
    shadow_module: Incomplete
    dequant: Incomplete
    logger: Incomplete
    def __init__(self, q_module, float_module, logger_cls) -> None: ...
    def forward(self, *x) -> torch.Tensor:
        """
        """
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        """
    def add_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        """
        """
    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        """
    def mul_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        """
        """
    def cat(self, x: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """
        """
    def add_relu(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        """

def prepare_model_with_stubs(float_module: nn.Module, q_module: nn.Module, module_swap_list: set[type], logger_cls: Callable) -> None:
    """Prepare the model by attaching the float module to its matching quantized
    module as the shadow if the float module type is in module_swap_list.

    Example usage::

        prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)
        q_model(data)
        ob_dict = get_logger_dict(q_model)

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        module_swap_list: list of float module types to attach the shadow
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
def _is_identical_module_type(mod1, mod2): ...
def compare_model_stub(float_model: nn.Module, q_model: nn.Module, module_swap_list: set[type], *data, logger_cls=...) -> dict[str, dict]:
    '''Compare quantized module in a model with its floating point counterpart,
    feeding both of them the same input. Return a dict with key corresponding to
    module names and each entry being a dictionary with two keys \'float\' and
    \'quantized\', containing the output tensors of quantized and its matching
    float shadow module. This dict can be used to compare and compute the module
    level quantization error.

    This function first call prepare_model_with_stubs() to swap the quantized
    module that we want to compare with the Shadow module, which takes quantized
    module, corresponding float module and logger as input, and creates a forward
    path inside to make the float module to shadow quantized module sharing the
    same input. The logger can be customizable, default logger is ShadowLogger
    and it will save the outputs of the quantized module and float module that
    can be used to compute the module level quantization error.

    Example usage::

        module_swap_list = [
            torchvision.models.quantization.resnet.QuantizableBasicBlock
        ]
        ob_dict = compare_model_stub(float_model, qmodel, module_swap_list, data)
        for key in ob_dict:
            print(
                key,
                compute_error(
                    ob_dict[key]["float"], ob_dict[key]["quantized"].dequantize()
                ),
            )

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        module_swap_list: list of float module types at which shadow modules will
            be attached.
        data: input data used to run the prepared q_model
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    '''
def get_matching_activations(float_module: nn.Module, q_module: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Find the matching activation between float and quantized modules.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module

    Return:
        act_dict: dict with key corresponding to quantized module names and each
        entry being a dictionary with two keys 'float' and 'quantized', containing
        the matching float and quantized activations
    """
def prepare_model_outputs(float_module: nn.Module, q_module: nn.Module, logger_cls=..., allow_list=None) -> None:
    """Prepare the model by attaching the logger to both float module
    and quantized module if they are in the allow_list.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger
    """
def compare_model_outputs(float_model: nn.Module, q_model: nn.Module, *data, logger_cls=..., allow_list=None) -> dict[str, dict[str, torch.Tensor]]:
    '''Compare output activations between float and quantized models at
    corresponding locations for the same input. Return a dict with key corresponding
    to quantized module names and each entry being a dictionary with two keys
    \'float\' and \'quantized\', containing the activations of quantized model and
    float model at matching locations. This dict can be used to compare and
    compute the propagation quantization error.

    Example usage::

        act_compare_dict = compare_model_outputs(float_model, qmodel, data)
        for key in act_compare_dict:
            print(
                key,
                compute_error(
                    act_compare_dict[key]["float"],
                    act_compare_dict[key]["quantized"].dequantize(),
                ),
            )

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        data: input data used to run the prepared float_model and q_model
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger

    Return:
        act_compare_dict: dict with key corresponding to quantized module names
        and each entry being a dictionary with two keys \'float\' and \'quantized\',
        containing the matching float and quantized activations
    '''
