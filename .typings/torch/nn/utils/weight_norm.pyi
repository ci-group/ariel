from torch.nn.modules import Module
from typing import Any, TypeVar

__all__ = ['WeightNorm', 'weight_norm', 'remove_weight_norm']

class WeightNorm:
    name: str
    dim: int
    def __init__(self, name: str, dim: int) -> None: ...
    def compute_weight(self, module: Module) -> Any: ...
    @staticmethod
    def apply(module, name: str, dim: int) -> WeightNorm: ...
    def remove(self, module: Module) -> None: ...
    def __call__(self, module: Module, inputs: Any) -> None: ...
T_module = TypeVar('T_module', bound=Module)

def weight_norm(module: T_module, name: str = 'weight', dim: int = 0) -> T_module:
    """Apply weight normalization to a parameter in the given module.

    .. math::
         \\mathbf{w} = g \\dfrac{\\mathbf{v}}{\\|\\mathbf{v}\\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    .. warning::

        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`
        which uses the modern parametrization API.  The new ``weight_norm`` is compatible
        with ``state_dict`` generated from old ``weight_norm``.

        Migration guide:

        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
          respectively.  If this is bothering you, please comment on
          https://github.com/pytorch/pytorch/issues/102999

        * To remove the weight normalization reparametrization, use
          :func:`torch.nn.utils.parametrize.remove_parametrizations`.

        * The weight is no longer recomputed once at module forward; instead, it will
          be recomputed on every access.  To restore the old behavior, use
          :func:`torch.nn.utils.parametrize.cached` before invoking the module
          in question.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
def remove_weight_norm(module: T_module, name: str = 'weight') -> T_module:
    """Remove the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
