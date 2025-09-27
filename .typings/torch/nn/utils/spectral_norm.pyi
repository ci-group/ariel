import torch
from _typeshed import Incomplete
from torch.nn.modules import Module
from typing import Any, TypeVar

__all__ = ['SpectralNorm', 'SpectralNormLoadStateDictPreHook', 'SpectralNormStateDictHook', 'spectral_norm', 'remove_spectral_norm']

class SpectralNorm:
    _version: int
    name: str
    dim: int
    n_power_iterations: int
    eps: float
    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12) -> None: ...
    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor: ...
    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor: ...
    def remove(self, module: Module) -> None: ...
    def __call__(self, module: Module, inputs: Any) -> None: ...
    def _solve_v_and_rescale(self, weight_mat, u, target_sigma): ...
    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> SpectralNorm: ...

class SpectralNormLoadStateDictPreHook:
    fn: Incomplete
    def __init__(self, fn) -> None: ...
    def __call__(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...

class SpectralNormStateDictHook:
    fn: Incomplete
    def __init__(self, fn) -> None: ...
    def __call__(self, module, state_dict, prefix, local_metadata) -> None: ...
T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module, name: str = 'weight', n_power_iterations: int = 1, eps: float = 1e-12, dim: int | None = None) -> T_module:
    """Apply spectral normalization to a parameter in the given module.

    .. math::
        \\mathbf{W}_{SN} = \\dfrac{\\mathbf{W}}{\\sigma(\\mathbf{W})},
        \\sigma(\\mathbf{W}) = \\max_{\\mathbf{h}: \\mathbf{h} \\ne 0} \\dfrac{\\|\\mathbf{W} \\mathbf{h}\\|_2}{\\|\\mathbf{h}\\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
def remove_spectral_norm(module: T_module, name: str = 'weight') -> T_module:
    """Remove the spectral normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
