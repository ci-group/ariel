import abc
from .match_utils import MatchAllNode as MatchAllNode, apply_match as apply_match
from .parametrization import BiasHook as BiasHook, FakeStructuredSparsity as FakeStructuredSparsity, module_contains_param as module_contains_param
from .prune_functions import prune_conv2d as prune_conv2d, prune_conv2d_activation_conv2d as prune_conv2d_activation_conv2d, prune_conv2d_activation_pool_conv2d as prune_conv2d_activation_pool_conv2d, prune_conv2d_conv2d as prune_conv2d_conv2d, prune_conv2d_pool_activation_conv2d as prune_conv2d_pool_activation_conv2d, prune_conv2d_pool_flatten_linear as prune_conv2d_pool_flatten_linear, prune_linear as prune_linear, prune_linear_activation_linear as prune_linear_activation_linear, prune_linear_linear as prune_linear_linear, prune_lstm_output_layernorm_linear as prune_lstm_output_layernorm_linear, prune_lstm_output_linear as prune_lstm_output_linear
from _typeshed import Incomplete
from torch import nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier as BaseSparsifier
from torch.fx import symbolic_trace as symbolic_trace
from torch.nn.utils import parametrize as parametrize
from typing import Callable

def _get_supported_structured_pruning_modules(): ...
def _get_supported_activation_functions(): ...
def _get_supported_activation_modules(): ...
def _get_default_structured_pruning_patterns() -> dict[tuple[type[nn.Module] | Callable | MatchAllNode | str, ...], Callable[..., None]]:
    """
    Returns the patterns for conv2d / linear conversion for each element in the activation functions/modules defined above.
    """

class BaseStructuredSparsifier(BaseSparsifier, metaclass=abc.ABCMeta):
    """Base class for structured pruning.

    Abstract methods that need to be implemented:
        - update_mask: Function to compute a new mask for all keys in the
            `groups` attribute.

    Args:
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    """
    patterns: Incomplete
    def __init__(self, defaults, patterns=None) -> None: ...
    def make_config_from_model(self, model: nn.Module, SUPPORTED_MODULES: set[type] | None = None) -> None: ...
    def _prepare(self, *args, **kwargs) -> None:
        """This function will attach the FakeStructuredSparsity parameterizations
        and BiasHooks at the appropriate points in the model.
        """
    traced: Incomplete
    def prune(self) -> None:
        """
        This function will FX symbolically trace the model and then find instances of the patterns
        defined in self.patterns (by default SUPPORTED_STRUCTURED_PRUNING_PATTERNS ).

        For each pattern, it will apply to corresponding conversion function, which will modify the output
        and input size expected by the modules within the pattern
        """
