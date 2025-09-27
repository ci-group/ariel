_is_fastpath_enabled: bool

def get_fastpath_enabled() -> bool:
    """Returns whether fast path for TransformerEncoder and MultiHeadAttention
    is enabled, or ``True`` if jit is scripting.

    .. note::
        The fastpath might not be run even if ``get_fastpath_enabled`` returns
        ``True`` unless all conditions on inputs are met.
    """
def set_fastpath_enabled(value: bool) -> None:
    """Sets whether fast path is enabled"""
