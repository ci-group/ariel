from torch.fx.node import Argument as Argument

def friendly_debug_info(v: object) -> Argument:
    """
    Helper function to print out debug info in a friendly way.
    """
def map_debug_info(a: Argument) -> Argument:
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
