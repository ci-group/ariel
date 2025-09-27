__all__ = ['visualize_sharding']

Color = tuple[float, float, float]

def visualize_sharding(dtensor, header: str = '', use_rich: bool = False):
    """
    Visualizes sharding in the terminal for :class:`DTensor` that are 1D or 2D.

    .. note:: This requires the ``tabulate`` package, or ``rich`` and ``matplotlib``.
              No sharding info will be printed for empty tensors
    """
