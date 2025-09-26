import typing as tp

X = tp.TypeVar('X')

class Registry(tp.MutableMapping[str, X]):
    """Registers function or classes as a dict."""
    data: dict[str, X]
    _information: dict[str, dict[tp.Hashable, tp.Any]]
    def __init__(self) -> None: ...
    def register(self, obj: X, info: dict[tp.Hashable, tp.Any] | None = None) -> X:
        """Decorator method for registering functions/classes
        The info variable can be filled up using the register_with_info
        decorator instead of this one.
        """
    def register_name(self, name: str, obj: X, info: dict[tp.Hashable, tp.Any] | None = None) -> None:
        """Register an object with a provided name"""
    def unregister(self, name: str) -> None:
        """Remove a previously-registered function or class, e.g. so you can
        re-register it in a Jupyter notebook.
        """
    def register_with_info(self, **info: tp.Any) -> tp.Callable[[X], X]:
        """Decorator for registering a function and information about it"""
    def get_info(self, name: str) -> dict[tp.Hashable, tp.Any]: ...
    def __getitem__(self, key: str) -> X: ...
    def __setitem__(self, key: str, value: X) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __iter__(self) -> tp.Iterator[str]: ...
    def __len__(self) -> int: ...
