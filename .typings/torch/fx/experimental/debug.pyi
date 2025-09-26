import torch.fx as fx

__all__ = ['set_trace']

def set_trace(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Sets a breakpoint in `gm`'s generated python code. It drops into pdb when
    `gm` gets run.

    Args:
        gm: graph module to insert breakpoint. It is then recompiled for it to
            take effect.

    Returns:
        the `gm` with breakpoint inserted.
    """
