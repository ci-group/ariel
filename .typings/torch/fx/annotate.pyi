from ._compatibility import compatibility as compatibility
from torch.fx.proxy import Proxy as Proxy

def annotate(val, type):
    """
    Annotates a Proxy object with a given type.

    This function annotates a val with a given type if a type of the val is a torch.fx.Proxy object
    Args:
        val (object): An object to be annotated if its type is torch.fx.Proxy.
        type (object): A type to be assigned to a given proxy object as val.
    Returns:
        The given val.
    Raises:
        RuntimeError: If a val already has a type in its node.
    """
