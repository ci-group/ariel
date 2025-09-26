from _typeshed import Incomplete
from torch._C import _get_backcompat_broadcast_warn as _get_backcompat_broadcast_warn, _get_backcompat_keepdim_warn as _get_backcompat_keepdim_warn, _set_backcompat_broadcast_warn as _set_backcompat_broadcast_warn, _set_backcompat_keepdim_warn as _set_backcompat_keepdim_warn

class Warning:
    setter: Incomplete
    getter: Incomplete
    def __init__(self, setter, getter) -> None: ...
    def set_enabled(self, value) -> None: ...
    def get_enabled(self): ...
    enabled: Incomplete

broadcast_warning: Incomplete
keepdim_warning: Incomplete
