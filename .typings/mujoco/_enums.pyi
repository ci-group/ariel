from typing import ClassVar, overload

class mjtAlignFree:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjALIGNFREE_AUTO: ClassVar[mjtAlignFree] = ...
    mjALIGNFREE_FALSE: ClassVar[mjtAlignFree] = ...
    mjALIGNFREE_TRUE: ClassVar[mjtAlignFree] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtAlignFree, value: int) -> None

        2. __init__(self: mujoco._enums.mjtAlignFree, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtAlignFree, value: int) -> None

        2. __init__(self: mujoco._enums.mjtAlignFree, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtAlignFree) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtAlignFree) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtAlignFree) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtAlignFree, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtAlignFree, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtBias:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjBIAS_AFFINE: ClassVar[mjtBias] = ...
    mjBIAS_MUSCLE: ClassVar[mjtBias] = ...
    mjBIAS_NONE: ClassVar[mjtBias] = ...
    mjBIAS_USER: ClassVar[mjtBias] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtBias, value: int) -> None

        2. __init__(self: mujoco._enums.mjtBias, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtBias, value: int) -> None

        2. __init__(self: mujoco._enums.mjtBias, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtBias, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtBias) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtBias) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtBias, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtBias) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtBias, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtBias, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtBias, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtBias, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtBias, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtBias, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtBias, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtBias, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtBias, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtBias, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtBuiltin:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjBUILTIN_CHECKER: ClassVar[mjtBuiltin] = ...
    mjBUILTIN_FLAT: ClassVar[mjtBuiltin] = ...
    mjBUILTIN_GRADIENT: ClassVar[mjtBuiltin] = ...
    mjBUILTIN_NONE: ClassVar[mjtBuiltin] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtBuiltin, value: int) -> None

        2. __init__(self: mujoco._enums.mjtBuiltin, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtBuiltin, value: int) -> None

        2. __init__(self: mujoco._enums.mjtBuiltin, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtBuiltin) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtBuiltin) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtBuiltin) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtBuiltin, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtBuiltin, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtButton:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjBUTTON_LEFT: ClassVar[mjtButton] = ...
    mjBUTTON_MIDDLE: ClassVar[mjtButton] = ...
    mjBUTTON_NONE: ClassVar[mjtButton] = ...
    mjBUTTON_RIGHT: ClassVar[mjtButton] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtButton, value: int) -> None

        2. __init__(self: mujoco._enums.mjtButton, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtButton, value: int) -> None

        2. __init__(self: mujoco._enums.mjtButton, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtButton, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtButton) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtButton) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtButton, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtButton) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtButton, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtButton, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtButton, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtButton, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtButton, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtButton, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtButton, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtButton, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtButton, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtButton, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtCamLight:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjCAMLIGHT_FIXED: ClassVar[mjtCamLight] = ...
    mjCAMLIGHT_TARGETBODY: ClassVar[mjtCamLight] = ...
    mjCAMLIGHT_TARGETBODYCOM: ClassVar[mjtCamLight] = ...
    mjCAMLIGHT_TRACK: ClassVar[mjtCamLight] = ...
    mjCAMLIGHT_TRACKCOM: ClassVar[mjtCamLight] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtCamLight, value: int) -> None

        2. __init__(self: mujoco._enums.mjtCamLight, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtCamLight, value: int) -> None

        2. __init__(self: mujoco._enums.mjtCamLight, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtCamLight, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtCamLight) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtCamLight) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtCamLight, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtCamLight) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtCamLight, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtCamLight, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtCamLight, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtCamLight, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtCamLight, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtCamLight, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtCamLight, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtCamLight, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtCamLight, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtCamLight, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtCamera:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjCAMERA_FIXED: ClassVar[mjtCamera] = ...
    mjCAMERA_FREE: ClassVar[mjtCamera] = ...
    mjCAMERA_TRACKING: ClassVar[mjtCamera] = ...
    mjCAMERA_USER: ClassVar[mjtCamera] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtCamera, value: int) -> None

        2. __init__(self: mujoco._enums.mjtCamera, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtCamera, value: int) -> None

        2. __init__(self: mujoco._enums.mjtCamera, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtCamera, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtCamera) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtCamera) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtCamera, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtCamera) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtCamera, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtCamera, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtCamera, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtCamera, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtCamera, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtCamera, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtCamera, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtCamera, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtCamera, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtCamera, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtCatBit:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjCAT_ALL: ClassVar[mjtCatBit] = ...
    mjCAT_DECOR: ClassVar[mjtCatBit] = ...
    mjCAT_DYNAMIC: ClassVar[mjtCatBit] = ...
    mjCAT_STATIC: ClassVar[mjtCatBit] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtCatBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtCatBit, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtCatBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtCatBit, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtCatBit, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtCatBit) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtCatBit) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtCatBit, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtCatBit) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtCatBit, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtCatBit, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtCatBit, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtCatBit, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtCatBit, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtCatBit, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtCatBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtCatBit, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtCatBit, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtCatBit, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtCone:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjCONE_ELLIPTIC: ClassVar[mjtCone] = ...
    mjCONE_PYRAMIDAL: ClassVar[mjtCone] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtCone, value: int) -> None

        2. __init__(self: mujoco._enums.mjtCone, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtCone, value: int) -> None

        2. __init__(self: mujoco._enums.mjtCone, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtCone, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtCone) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtCone) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtCone, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtCone) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtCone, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtCone, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtCone, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtCone, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtCone, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtCone, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtCone, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtCone, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtCone, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtCone, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtConstraint:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjCNSTR_CONTACT_ELLIPTIC: ClassVar[mjtConstraint] = ...
    mjCNSTR_CONTACT_FRICTIONLESS: ClassVar[mjtConstraint] = ...
    mjCNSTR_CONTACT_PYRAMIDAL: ClassVar[mjtConstraint] = ...
    mjCNSTR_EQUALITY: ClassVar[mjtConstraint] = ...
    mjCNSTR_FRICTION_DOF: ClassVar[mjtConstraint] = ...
    mjCNSTR_FRICTION_TENDON: ClassVar[mjtConstraint] = ...
    mjCNSTR_LIMIT_JOINT: ClassVar[mjtConstraint] = ...
    mjCNSTR_LIMIT_TENDON: ClassVar[mjtConstraint] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtConstraint, value: int) -> None

        2. __init__(self: mujoco._enums.mjtConstraint, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtConstraint, value: int) -> None

        2. __init__(self: mujoco._enums.mjtConstraint, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtConstraint, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtConstraint) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtConstraint) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtConstraint, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtConstraint) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtConstraint, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtConstraint, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtConstraint, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtConstraint, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtConstraint, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtConstraint, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtConstraint, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtConstraint, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtConstraint, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtConstraint, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtConstraintState:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjCNSTRSTATE_CONE: ClassVar[mjtConstraintState] = ...
    mjCNSTRSTATE_LINEARNEG: ClassVar[mjtConstraintState] = ...
    mjCNSTRSTATE_LINEARPOS: ClassVar[mjtConstraintState] = ...
    mjCNSTRSTATE_QUADRATIC: ClassVar[mjtConstraintState] = ...
    mjCNSTRSTATE_SATISFIED: ClassVar[mjtConstraintState] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtConstraintState, value: int) -> None

        2. __init__(self: mujoco._enums.mjtConstraintState, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtConstraintState, value: int) -> None

        2. __init__(self: mujoco._enums.mjtConstraintState, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtConstraintState) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtConstraintState) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtConstraintState) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtConstraintState, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtConstraintState, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtDataType:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjDATATYPE_AXIS: ClassVar[mjtDataType] = ...
    mjDATATYPE_POSITIVE: ClassVar[mjtDataType] = ...
    mjDATATYPE_QUATERNION: ClassVar[mjtDataType] = ...
    mjDATATYPE_REAL: ClassVar[mjtDataType] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtDataType, value: int) -> None

        2. __init__(self: mujoco._enums.mjtDataType, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtDataType, value: int) -> None

        2. __init__(self: mujoco._enums.mjtDataType, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtDataType, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtDataType) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtDataType) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtDataType, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtDataType) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtDataType, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtDataType, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtDataType, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtDataType, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtDataType, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtDataType, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtDataType, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtDataType, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtDataType, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtDataType, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtDepthMap:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjDEPTH_ZEROFAR: ClassVar[mjtDepthMap] = ...
    mjDEPTH_ZERONEAR: ClassVar[mjtDepthMap] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtDepthMap, value: int) -> None

        2. __init__(self: mujoco._enums.mjtDepthMap, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtDepthMap, value: int) -> None

        2. __init__(self: mujoco._enums.mjtDepthMap, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtDepthMap) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtDepthMap) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtDepthMap) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtDepthMap, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtDepthMap, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtDisableBit:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjDSBL_ACTUATION: ClassVar[mjtDisableBit] = ...
    mjDSBL_AUTORESET: ClassVar[mjtDisableBit] = ...
    mjDSBL_CLAMPCTRL: ClassVar[mjtDisableBit] = ...
    mjDSBL_CONSTRAINT: ClassVar[mjtDisableBit] = ...
    mjDSBL_CONTACT: ClassVar[mjtDisableBit] = ...
    mjDSBL_EQUALITY: ClassVar[mjtDisableBit] = ...
    mjDSBL_EULERDAMP: ClassVar[mjtDisableBit] = ...
    mjDSBL_FILTERPARENT: ClassVar[mjtDisableBit] = ...
    mjDSBL_FRICTIONLOSS: ClassVar[mjtDisableBit] = ...
    mjDSBL_GRAVITY: ClassVar[mjtDisableBit] = ...
    mjDSBL_LIMIT: ClassVar[mjtDisableBit] = ...
    mjDSBL_MIDPHASE: ClassVar[mjtDisableBit] = ...
    mjDSBL_NATIVECCD: ClassVar[mjtDisableBit] = ...
    mjDSBL_PASSIVE: ClassVar[mjtDisableBit] = ...
    mjDSBL_REFSAFE: ClassVar[mjtDisableBit] = ...
    mjDSBL_SENSOR: ClassVar[mjtDisableBit] = ...
    mjDSBL_WARMSTART: ClassVar[mjtDisableBit] = ...
    mjNDISABLE: ClassVar[mjtDisableBit] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtDisableBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtDisableBit, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtDisableBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtDisableBit, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtDisableBit) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtDisableBit) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtDisableBit) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtDisableBit, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtDisableBit, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtDyn:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjDYN_FILTER: ClassVar[mjtDyn] = ...
    mjDYN_FILTEREXACT: ClassVar[mjtDyn] = ...
    mjDYN_INTEGRATOR: ClassVar[mjtDyn] = ...
    mjDYN_MUSCLE: ClassVar[mjtDyn] = ...
    mjDYN_NONE: ClassVar[mjtDyn] = ...
    mjDYN_USER: ClassVar[mjtDyn] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtDyn, value: int) -> None

        2. __init__(self: mujoco._enums.mjtDyn, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtDyn, value: int) -> None

        2. __init__(self: mujoco._enums.mjtDyn, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtDyn, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtDyn) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtDyn) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtDyn, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtDyn) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtDyn, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtDyn, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtDyn, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtDyn, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtDyn, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtDyn, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtDyn, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtDyn, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtDyn, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtDyn, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtEnableBit:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjENBL_ENERGY: ClassVar[mjtEnableBit] = ...
    mjENBL_FWDINV: ClassVar[mjtEnableBit] = ...
    mjENBL_INVDISCRETE: ClassVar[mjtEnableBit] = ...
    mjENBL_ISLAND: ClassVar[mjtEnableBit] = ...
    mjENBL_MULTICCD: ClassVar[mjtEnableBit] = ...
    mjENBL_OVERRIDE: ClassVar[mjtEnableBit] = ...
    mjNENABLE: ClassVar[mjtEnableBit] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtEnableBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtEnableBit, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtEnableBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtEnableBit, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtEnableBit) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtEnableBit) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtEnableBit) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtEnableBit, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtEnableBit, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtEq:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjEQ_CONNECT: ClassVar[mjtEq] = ...
    mjEQ_DISTANCE: ClassVar[mjtEq] = ...
    mjEQ_FLEX: ClassVar[mjtEq] = ...
    mjEQ_JOINT: ClassVar[mjtEq] = ...
    mjEQ_TENDON: ClassVar[mjtEq] = ...
    mjEQ_WELD: ClassVar[mjtEq] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtEq, value: int) -> None

        2. __init__(self: mujoco._enums.mjtEq, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtEq, value: int) -> None

        2. __init__(self: mujoco._enums.mjtEq, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtEq, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtEq) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtEq) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtEq, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtEq) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtEq, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtEq, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtEq, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtEq, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtEq, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtEq, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtEq, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtEq, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtEq, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtEq, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtEvent:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjEVENT_FILESDROP: ClassVar[mjtEvent] = ...
    mjEVENT_KEY: ClassVar[mjtEvent] = ...
    mjEVENT_MOVE: ClassVar[mjtEvent] = ...
    mjEVENT_NONE: ClassVar[mjtEvent] = ...
    mjEVENT_PRESS: ClassVar[mjtEvent] = ...
    mjEVENT_REDRAW: ClassVar[mjtEvent] = ...
    mjEVENT_RELEASE: ClassVar[mjtEvent] = ...
    mjEVENT_RESIZE: ClassVar[mjtEvent] = ...
    mjEVENT_SCROLL: ClassVar[mjtEvent] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtEvent, value: int) -> None

        2. __init__(self: mujoco._enums.mjtEvent, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtEvent, value: int) -> None

        2. __init__(self: mujoco._enums.mjtEvent, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtEvent, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtEvent) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtEvent) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtEvent, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtEvent) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtEvent, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtEvent, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtEvent, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtEvent, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtEvent, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtEvent, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtEvent, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtEvent, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtEvent, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtEvent, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtFlexSelf:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjFLEXSELF_AUTO: ClassVar[mjtFlexSelf] = ...
    mjFLEXSELF_BVH: ClassVar[mjtFlexSelf] = ...
    mjFLEXSELF_NARROW: ClassVar[mjtFlexSelf] = ...
    mjFLEXSELF_NONE: ClassVar[mjtFlexSelf] = ...
    mjFLEXSELF_SAP: ClassVar[mjtFlexSelf] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFlexSelf, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFlexSelf, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFlexSelf, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFlexSelf, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtFlexSelf) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtFlexSelf) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtFlexSelf) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtFlexSelf, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtFlexSelf, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtFont:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjFONT_BIG: ClassVar[mjtFont] = ...
    mjFONT_NORMAL: ClassVar[mjtFont] = ...
    mjFONT_SHADOW: ClassVar[mjtFont] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFont, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFont, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFont, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFont, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtFont, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtFont) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtFont) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtFont, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtFont) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtFont, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtFont, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtFont, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtFont, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtFont, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtFont, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFont, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFont, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtFont, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtFont, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtFontScale:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjFONTSCALE_100: ClassVar[mjtFontScale] = ...
    mjFONTSCALE_150: ClassVar[mjtFontScale] = ...
    mjFONTSCALE_200: ClassVar[mjtFontScale] = ...
    mjFONTSCALE_250: ClassVar[mjtFontScale] = ...
    mjFONTSCALE_300: ClassVar[mjtFontScale] = ...
    mjFONTSCALE_50: ClassVar[mjtFontScale] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFontScale, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFontScale, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFontScale, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFontScale, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtFontScale, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtFontScale) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtFontScale) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtFontScale, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtFontScale) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtFontScale, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtFontScale, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtFontScale, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtFontScale, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtFontScale, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtFontScale, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFontScale, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFontScale, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtFontScale, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtFontScale, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtFrame:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjFRAME_BODY: ClassVar[mjtFrame] = ...
    mjFRAME_CAMERA: ClassVar[mjtFrame] = ...
    mjFRAME_CONTACT: ClassVar[mjtFrame] = ...
    mjFRAME_GEOM: ClassVar[mjtFrame] = ...
    mjFRAME_LIGHT: ClassVar[mjtFrame] = ...
    mjFRAME_NONE: ClassVar[mjtFrame] = ...
    mjFRAME_SITE: ClassVar[mjtFrame] = ...
    mjFRAME_WORLD: ClassVar[mjtFrame] = ...
    mjNFRAME: ClassVar[mjtFrame] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFrame, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFrame, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFrame, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFrame, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtFrame, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtFrame) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtFrame) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtFrame, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtFrame) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtFrame, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtFrame, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtFrame, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtFrame, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtFrame, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtFrame, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFrame, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFrame, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtFrame, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtFrame, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtFramebuffer:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjFB_OFFSCREEN: ClassVar[mjtFramebuffer] = ...
    mjFB_WINDOW: ClassVar[mjtFramebuffer] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFramebuffer, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFramebuffer, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtFramebuffer, value: int) -> None

        2. __init__(self: mujoco._enums.mjtFramebuffer, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtFramebuffer) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtFramebuffer) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtFramebuffer) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtFramebuffer, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtFramebuffer, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtGain:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjGAIN_AFFINE: ClassVar[mjtGain] = ...
    mjGAIN_FIXED: ClassVar[mjtGain] = ...
    mjGAIN_MUSCLE: ClassVar[mjtGain] = ...
    mjGAIN_USER: ClassVar[mjtGain] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtGain, value: int) -> None

        2. __init__(self: mujoco._enums.mjtGain, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtGain, value: int) -> None

        2. __init__(self: mujoco._enums.mjtGain, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtGain, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtGain) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtGain) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtGain, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtGain) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtGain, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtGain, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtGain, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtGain, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtGain, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtGain, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtGain, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtGain, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtGain, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtGain, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtGeom:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjGEOM_ARROW: ClassVar[mjtGeom] = ...
    mjGEOM_ARROW1: ClassVar[mjtGeom] = ...
    mjGEOM_ARROW2: ClassVar[mjtGeom] = ...
    mjGEOM_BOX: ClassVar[mjtGeom] = ...
    mjGEOM_CAPSULE: ClassVar[mjtGeom] = ...
    mjGEOM_CYLINDER: ClassVar[mjtGeom] = ...
    mjGEOM_ELLIPSOID: ClassVar[mjtGeom] = ...
    mjGEOM_FLEX: ClassVar[mjtGeom] = ...
    mjGEOM_HFIELD: ClassVar[mjtGeom] = ...
    mjGEOM_LABEL: ClassVar[mjtGeom] = ...
    mjGEOM_LINE: ClassVar[mjtGeom] = ...
    mjGEOM_LINEBOX: ClassVar[mjtGeom] = ...
    mjGEOM_MESH: ClassVar[mjtGeom] = ...
    mjGEOM_NONE: ClassVar[mjtGeom] = ...
    mjGEOM_PLANE: ClassVar[mjtGeom] = ...
    mjGEOM_SDF: ClassVar[mjtGeom] = ...
    mjGEOM_SKIN: ClassVar[mjtGeom] = ...
    mjGEOM_SPHERE: ClassVar[mjtGeom] = ...
    mjGEOM_TRIANGLE: ClassVar[mjtGeom] = ...
    mjNGEOMTYPES: ClassVar[mjtGeom] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtGeom, value: int) -> None

        2. __init__(self: mujoco._enums.mjtGeom, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtGeom, value: int) -> None

        2. __init__(self: mujoco._enums.mjtGeom, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtGeom, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtGeom) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtGeom) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtGeom, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtGeom) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtGeom, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtGeom, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtGeom, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtGeom, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtGeom, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtGeom, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtGeom, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtGeom, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtGeom, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtGeom, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtGeomInertia:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjINERTIA_SHELL: ClassVar[mjtGeomInertia] = ...
    mjINERTIA_VOLUME: ClassVar[mjtGeomInertia] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtGeomInertia, value: int) -> None

        2. __init__(self: mujoco._enums.mjtGeomInertia, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtGeomInertia, value: int) -> None

        2. __init__(self: mujoco._enums.mjtGeomInertia, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtGeomInertia) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtGeomInertia) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtGeomInertia) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtGeomInertia, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtGeomInertia, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtGridPos:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjGRID_BOTTOM: ClassVar[mjtGridPos] = ...
    mjGRID_BOTTOMLEFT: ClassVar[mjtGridPos] = ...
    mjGRID_BOTTOMRIGHT: ClassVar[mjtGridPos] = ...
    mjGRID_LEFT: ClassVar[mjtGridPos] = ...
    mjGRID_RIGHT: ClassVar[mjtGridPos] = ...
    mjGRID_TOP: ClassVar[mjtGridPos] = ...
    mjGRID_TOPLEFT: ClassVar[mjtGridPos] = ...
    mjGRID_TOPRIGHT: ClassVar[mjtGridPos] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtGridPos, value: int) -> None

        2. __init__(self: mujoco._enums.mjtGridPos, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtGridPos, value: int) -> None

        2. __init__(self: mujoco._enums.mjtGridPos, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtGridPos, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtGridPos) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtGridPos) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtGridPos, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtGridPos) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtGridPos, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtGridPos, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtGridPos, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtGridPos, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtGridPos, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtGridPos, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtGridPos, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtGridPos, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtGridPos, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtGridPos, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtInertiaFromGeom:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjINERTIAFROMGEOM_AUTO: ClassVar[mjtInertiaFromGeom] = ...
    mjINERTIAFROMGEOM_FALSE: ClassVar[mjtInertiaFromGeom] = ...
    mjINERTIAFROMGEOM_TRUE: ClassVar[mjtInertiaFromGeom] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtInertiaFromGeom, value: int) -> None

        2. __init__(self: mujoco._enums.mjtInertiaFromGeom, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtInertiaFromGeom, value: int) -> None

        2. __init__(self: mujoco._enums.mjtInertiaFromGeom, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtInertiaFromGeom) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtInertiaFromGeom) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtInertiaFromGeom) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtInertiaFromGeom, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtInertiaFromGeom, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtIntegrator:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjINT_EULER: ClassVar[mjtIntegrator] = ...
    mjINT_IMPLICIT: ClassVar[mjtIntegrator] = ...
    mjINT_IMPLICITFAST: ClassVar[mjtIntegrator] = ...
    mjINT_RK4: ClassVar[mjtIntegrator] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtIntegrator, value: int) -> None

        2. __init__(self: mujoco._enums.mjtIntegrator, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtIntegrator, value: int) -> None

        2. __init__(self: mujoco._enums.mjtIntegrator, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtIntegrator) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtIntegrator) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtIntegrator) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtIntegrator, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtIntegrator, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtItem:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjITEM_BUTTON: ClassVar[mjtItem] = ...
    mjITEM_CHECKBYTE: ClassVar[mjtItem] = ...
    mjITEM_CHECKINT: ClassVar[mjtItem] = ...
    mjITEM_EDITFLOAT: ClassVar[mjtItem] = ...
    mjITEM_EDITINT: ClassVar[mjtItem] = ...
    mjITEM_EDITNUM: ClassVar[mjtItem] = ...
    mjITEM_EDITTXT: ClassVar[mjtItem] = ...
    mjITEM_END: ClassVar[mjtItem] = ...
    mjITEM_RADIO: ClassVar[mjtItem] = ...
    mjITEM_RADIOLINE: ClassVar[mjtItem] = ...
    mjITEM_SECTION: ClassVar[mjtItem] = ...
    mjITEM_SELECT: ClassVar[mjtItem] = ...
    mjITEM_SEPARATOR: ClassVar[mjtItem] = ...
    mjITEM_SLIDERINT: ClassVar[mjtItem] = ...
    mjITEM_SLIDERNUM: ClassVar[mjtItem] = ...
    mjITEM_STATIC: ClassVar[mjtItem] = ...
    mjNITEM: ClassVar[mjtItem] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtItem, value: int) -> None

        2. __init__(self: mujoco._enums.mjtItem, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtItem, value: int) -> None

        2. __init__(self: mujoco._enums.mjtItem, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtItem, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtItem) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtItem) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtItem, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtItem) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtItem, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtItem, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtItem, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtItem, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtItem, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtItem, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtItem, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtItem, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtItem, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtItem, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtJacobian:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjJAC_AUTO: ClassVar[mjtJacobian] = ...
    mjJAC_DENSE: ClassVar[mjtJacobian] = ...
    mjJAC_SPARSE: ClassVar[mjtJacobian] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtJacobian, value: int) -> None

        2. __init__(self: mujoco._enums.mjtJacobian, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtJacobian, value: int) -> None

        2. __init__(self: mujoco._enums.mjtJacobian, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtJacobian, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtJacobian) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtJacobian) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtJacobian, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtJacobian) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtJacobian, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtJacobian, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtJacobian, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtJacobian, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtJacobian, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtJacobian, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtJacobian, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtJacobian, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtJacobian, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtJacobian, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtJoint:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjJNT_BALL: ClassVar[mjtJoint] = ...
    mjJNT_FREE: ClassVar[mjtJoint] = ...
    mjJNT_HINGE: ClassVar[mjtJoint] = ...
    mjJNT_SLIDE: ClassVar[mjtJoint] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtJoint, value: int) -> None

        2. __init__(self: mujoco._enums.mjtJoint, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtJoint, value: int) -> None

        2. __init__(self: mujoco._enums.mjtJoint, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtJoint, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtJoint) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtJoint) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtJoint, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtJoint) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtJoint, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtJoint, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtJoint, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtJoint, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtJoint, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtJoint, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtJoint, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtJoint, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtJoint, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtJoint, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtLRMode:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjLRMODE_ALL: ClassVar[mjtLRMode] = ...
    mjLRMODE_MUSCLE: ClassVar[mjtLRMode] = ...
    mjLRMODE_MUSCLEUSER: ClassVar[mjtLRMode] = ...
    mjLRMODE_NONE: ClassVar[mjtLRMode] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtLRMode, value: int) -> None

        2. __init__(self: mujoco._enums.mjtLRMode, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtLRMode, value: int) -> None

        2. __init__(self: mujoco._enums.mjtLRMode, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtLRMode, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtLRMode) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtLRMode) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtLRMode, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtLRMode) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtLRMode, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtLRMode, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtLRMode, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtLRMode, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtLRMode, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtLRMode, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtLRMode, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtLRMode, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtLRMode, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtLRMode, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtLabel:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjLABEL_ACTUATOR: ClassVar[mjtLabel] = ...
    mjLABEL_BODY: ClassVar[mjtLabel] = ...
    mjLABEL_CAMERA: ClassVar[mjtLabel] = ...
    mjLABEL_CONSTRAINT: ClassVar[mjtLabel] = ...
    mjLABEL_CONTACTFORCE: ClassVar[mjtLabel] = ...
    mjLABEL_CONTACTPOINT: ClassVar[mjtLabel] = ...
    mjLABEL_FLEX: ClassVar[mjtLabel] = ...
    mjLABEL_GEOM: ClassVar[mjtLabel] = ...
    mjLABEL_ISLAND: ClassVar[mjtLabel] = ...
    mjLABEL_JOINT: ClassVar[mjtLabel] = ...
    mjLABEL_LIGHT: ClassVar[mjtLabel] = ...
    mjLABEL_NONE: ClassVar[mjtLabel] = ...
    mjLABEL_SELECTION: ClassVar[mjtLabel] = ...
    mjLABEL_SELPNT: ClassVar[mjtLabel] = ...
    mjLABEL_SITE: ClassVar[mjtLabel] = ...
    mjLABEL_SKIN: ClassVar[mjtLabel] = ...
    mjLABEL_TENDON: ClassVar[mjtLabel] = ...
    mjNLABEL: ClassVar[mjtLabel] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtLabel, value: int) -> None

        2. __init__(self: mujoco._enums.mjtLabel, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtLabel, value: int) -> None

        2. __init__(self: mujoco._enums.mjtLabel, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtLabel, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtLabel) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtLabel) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtLabel, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtLabel) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtLabel, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtLabel, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtLabel, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtLabel, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtLabel, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtLabel, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtLabel, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtLabel, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtLabel, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtLabel, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtLimited:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjLIMITED_AUTO: ClassVar[mjtLimited] = ...
    mjLIMITED_FALSE: ClassVar[mjtLimited] = ...
    mjLIMITED_TRUE: ClassVar[mjtLimited] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtLimited, value: int) -> None

        2. __init__(self: mujoco._enums.mjtLimited, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtLimited, value: int) -> None

        2. __init__(self: mujoco._enums.mjtLimited, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtLimited, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtLimited) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtLimited) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtLimited, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtLimited) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtLimited, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtLimited, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtLimited, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtLimited, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtLimited, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtLimited, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtLimited, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtLimited, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtLimited, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtLimited, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtMark:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjMARK_CROSS: ClassVar[mjtMark] = ...
    mjMARK_EDGE: ClassVar[mjtMark] = ...
    mjMARK_NONE: ClassVar[mjtMark] = ...
    mjMARK_RANDOM: ClassVar[mjtMark] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtMark, value: int) -> None

        2. __init__(self: mujoco._enums.mjtMark, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtMark, value: int) -> None

        2. __init__(self: mujoco._enums.mjtMark, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtMark, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtMark) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtMark) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtMark, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtMark) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtMark, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtMark, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtMark, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtMark, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtMark, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtMark, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtMark, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtMark, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtMark, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtMark, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtMeshInertia:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjMESH_INERTIA_CONVEX: ClassVar[mjtMeshInertia] = ...
    mjMESH_INERTIA_EXACT: ClassVar[mjtMeshInertia] = ...
    mjMESH_INERTIA_LEGACY: ClassVar[mjtMeshInertia] = ...
    mjMESH_INERTIA_SHELL: ClassVar[mjtMeshInertia] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtMeshInertia, value: int) -> None

        2. __init__(self: mujoco._enums.mjtMeshInertia, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtMeshInertia, value: int) -> None

        2. __init__(self: mujoco._enums.mjtMeshInertia, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtMeshInertia) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtMeshInertia) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtMeshInertia) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtMeshInertia, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtMeshInertia, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtMouse:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjMOUSE_MOVE_H: ClassVar[mjtMouse] = ...
    mjMOUSE_MOVE_V: ClassVar[mjtMouse] = ...
    mjMOUSE_NONE: ClassVar[mjtMouse] = ...
    mjMOUSE_ROTATE_H: ClassVar[mjtMouse] = ...
    mjMOUSE_ROTATE_V: ClassVar[mjtMouse] = ...
    mjMOUSE_SELECT: ClassVar[mjtMouse] = ...
    mjMOUSE_ZOOM: ClassVar[mjtMouse] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtMouse, value: int) -> None

        2. __init__(self: mujoco._enums.mjtMouse, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtMouse, value: int) -> None

        2. __init__(self: mujoco._enums.mjtMouse, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtMouse, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtMouse) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtMouse) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtMouse, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtMouse) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtMouse, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtMouse, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtMouse, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtMouse, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtMouse, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtMouse, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtMouse, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtMouse, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtMouse, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtMouse, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtObj:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjNOBJECT: ClassVar[mjtObj] = ...
    mjOBJ_ACTUATOR: ClassVar[mjtObj] = ...
    mjOBJ_BODY: ClassVar[mjtObj] = ...
    mjOBJ_CAMERA: ClassVar[mjtObj] = ...
    mjOBJ_DEFAULT: ClassVar[mjtObj] = ...
    mjOBJ_DOF: ClassVar[mjtObj] = ...
    mjOBJ_EQUALITY: ClassVar[mjtObj] = ...
    mjOBJ_EXCLUDE: ClassVar[mjtObj] = ...
    mjOBJ_FLEX: ClassVar[mjtObj] = ...
    mjOBJ_FRAME: ClassVar[mjtObj] = ...
    mjOBJ_GEOM: ClassVar[mjtObj] = ...
    mjOBJ_HFIELD: ClassVar[mjtObj] = ...
    mjOBJ_JOINT: ClassVar[mjtObj] = ...
    mjOBJ_KEY: ClassVar[mjtObj] = ...
    mjOBJ_LIGHT: ClassVar[mjtObj] = ...
    mjOBJ_MATERIAL: ClassVar[mjtObj] = ...
    mjOBJ_MESH: ClassVar[mjtObj] = ...
    mjOBJ_MODEL: ClassVar[mjtObj] = ...
    mjOBJ_NUMERIC: ClassVar[mjtObj] = ...
    mjOBJ_PAIR: ClassVar[mjtObj] = ...
    mjOBJ_PLUGIN: ClassVar[mjtObj] = ...
    mjOBJ_SENSOR: ClassVar[mjtObj] = ...
    mjOBJ_SITE: ClassVar[mjtObj] = ...
    mjOBJ_SKIN: ClassVar[mjtObj] = ...
    mjOBJ_TENDON: ClassVar[mjtObj] = ...
    mjOBJ_TEXT: ClassVar[mjtObj] = ...
    mjOBJ_TEXTURE: ClassVar[mjtObj] = ...
    mjOBJ_TUPLE: ClassVar[mjtObj] = ...
    mjOBJ_UNKNOWN: ClassVar[mjtObj] = ...
    mjOBJ_XBODY: ClassVar[mjtObj] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtObj, value: int) -> None

        2. __init__(self: mujoco._enums.mjtObj, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtObj, value: int) -> None

        2. __init__(self: mujoco._enums.mjtObj, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtObj, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtObj) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtObj) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtObj, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtObj) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtObj, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtObj, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtObj, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtObj, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtObj, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtObj, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtObj, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtObj, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtObj, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtObj, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtOrientation:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjORIENTATION_AXISANGLE: ClassVar[mjtOrientation] = ...
    mjORIENTATION_EULER: ClassVar[mjtOrientation] = ...
    mjORIENTATION_QUAT: ClassVar[mjtOrientation] = ...
    mjORIENTATION_XYAXES: ClassVar[mjtOrientation] = ...
    mjORIENTATION_ZAXIS: ClassVar[mjtOrientation] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtOrientation, value: int) -> None

        2. __init__(self: mujoco._enums.mjtOrientation, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtOrientation, value: int) -> None

        2. __init__(self: mujoco._enums.mjtOrientation, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtOrientation, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtOrientation) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtOrientation) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtOrientation, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtOrientation) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtOrientation, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtOrientation, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtOrientation, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtOrientation, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtOrientation, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtOrientation, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtOrientation, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtOrientation, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtOrientation, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtOrientation, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtPertBit:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjPERT_ROTATE: ClassVar[mjtPertBit] = ...
    mjPERT_TRANSLATE: ClassVar[mjtPertBit] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtPertBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtPertBit, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtPertBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtPertBit, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtPertBit, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtPertBit) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtPertBit) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtPertBit, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtPertBit) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtPertBit, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtPertBit, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtPertBit, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtPertBit, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtPertBit, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtPertBit, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtPertBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtPertBit, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtPertBit, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtPertBit, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtPluginCapabilityBit:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjPLUGIN_ACTUATOR: ClassVar[mjtPluginCapabilityBit] = ...
    mjPLUGIN_PASSIVE: ClassVar[mjtPluginCapabilityBit] = ...
    mjPLUGIN_SDF: ClassVar[mjtPluginCapabilityBit] = ...
    mjPLUGIN_SENSOR: ClassVar[mjtPluginCapabilityBit] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtPluginCapabilityBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtPluginCapabilityBit, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtPluginCapabilityBit, value: int) -> None

        2. __init__(self: mujoco._enums.mjtPluginCapabilityBit, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtPluginCapabilityBit) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtPluginCapabilityBit) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtPluginCapabilityBit) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtPluginCapabilityBit, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtRndFlag:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjNRNDFLAG: ClassVar[mjtRndFlag] = ...
    mjRND_ADDITIVE: ClassVar[mjtRndFlag] = ...
    mjRND_CULL_FACE: ClassVar[mjtRndFlag] = ...
    mjRND_FOG: ClassVar[mjtRndFlag] = ...
    mjRND_HAZE: ClassVar[mjtRndFlag] = ...
    mjRND_IDCOLOR: ClassVar[mjtRndFlag] = ...
    mjRND_REFLECTION: ClassVar[mjtRndFlag] = ...
    mjRND_SEGMENT: ClassVar[mjtRndFlag] = ...
    mjRND_SHADOW: ClassVar[mjtRndFlag] = ...
    mjRND_SKYBOX: ClassVar[mjtRndFlag] = ...
    mjRND_WIREFRAME: ClassVar[mjtRndFlag] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtRndFlag, value: int) -> None

        2. __init__(self: mujoco._enums.mjtRndFlag, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtRndFlag, value: int) -> None

        2. __init__(self: mujoco._enums.mjtRndFlag, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtRndFlag) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtRndFlag) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtRndFlag) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtRndFlag, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtRndFlag, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtSameFrame:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjSAMEFRAME_BODY: ClassVar[mjtSameFrame] = ...
    mjSAMEFRAME_BODYROT: ClassVar[mjtSameFrame] = ...
    mjSAMEFRAME_INERTIA: ClassVar[mjtSameFrame] = ...
    mjSAMEFRAME_INERTIAROT: ClassVar[mjtSameFrame] = ...
    mjSAMEFRAME_NONE: ClassVar[mjtSameFrame] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtSameFrame, value: int) -> None

        2. __init__(self: mujoco._enums.mjtSameFrame, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtSameFrame, value: int) -> None

        2. __init__(self: mujoco._enums.mjtSameFrame, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtSameFrame) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtSameFrame) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtSameFrame) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtSameFrame, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtSameFrame, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtSection:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjSECT_CLOSED: ClassVar[mjtSection] = ...
    mjSECT_FIXED: ClassVar[mjtSection] = ...
    mjSECT_OPEN: ClassVar[mjtSection] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtSection, value: int) -> None

        2. __init__(self: mujoco._enums.mjtSection, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtSection, value: int) -> None

        2. __init__(self: mujoco._enums.mjtSection, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtSection, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtSection) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtSection) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtSection, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtSection) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtSection, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtSection, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtSection, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtSection, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtSection, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtSection, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtSection, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtSection, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtSection, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtSection, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtSensor:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjSENS_ACCELEROMETER: ClassVar[mjtSensor] = ...
    mjSENS_ACTUATORFRC: ClassVar[mjtSensor] = ...
    mjSENS_ACTUATORPOS: ClassVar[mjtSensor] = ...
    mjSENS_ACTUATORVEL: ClassVar[mjtSensor] = ...
    mjSENS_BALLANGVEL: ClassVar[mjtSensor] = ...
    mjSENS_BALLQUAT: ClassVar[mjtSensor] = ...
    mjSENS_CAMPROJECTION: ClassVar[mjtSensor] = ...
    mjSENS_CLOCK: ClassVar[mjtSensor] = ...
    mjSENS_E_KINETIC: ClassVar[mjtSensor] = ...
    mjSENS_E_POTENTIAL: ClassVar[mjtSensor] = ...
    mjSENS_FORCE: ClassVar[mjtSensor] = ...
    mjSENS_FRAMEANGACC: ClassVar[mjtSensor] = ...
    mjSENS_FRAMEANGVEL: ClassVar[mjtSensor] = ...
    mjSENS_FRAMELINACC: ClassVar[mjtSensor] = ...
    mjSENS_FRAMELINVEL: ClassVar[mjtSensor] = ...
    mjSENS_FRAMEPOS: ClassVar[mjtSensor] = ...
    mjSENS_FRAMEQUAT: ClassVar[mjtSensor] = ...
    mjSENS_FRAMEXAXIS: ClassVar[mjtSensor] = ...
    mjSENS_FRAMEYAXIS: ClassVar[mjtSensor] = ...
    mjSENS_FRAMEZAXIS: ClassVar[mjtSensor] = ...
    mjSENS_GEOMDIST: ClassVar[mjtSensor] = ...
    mjSENS_GEOMFROMTO: ClassVar[mjtSensor] = ...
    mjSENS_GEOMNORMAL: ClassVar[mjtSensor] = ...
    mjSENS_GYRO: ClassVar[mjtSensor] = ...
    mjSENS_JOINTACTFRC: ClassVar[mjtSensor] = ...
    mjSENS_JOINTLIMITFRC: ClassVar[mjtSensor] = ...
    mjSENS_JOINTLIMITPOS: ClassVar[mjtSensor] = ...
    mjSENS_JOINTLIMITVEL: ClassVar[mjtSensor] = ...
    mjSENS_JOINTPOS: ClassVar[mjtSensor] = ...
    mjSENS_JOINTVEL: ClassVar[mjtSensor] = ...
    mjSENS_MAGNETOMETER: ClassVar[mjtSensor] = ...
    mjSENS_PLUGIN: ClassVar[mjtSensor] = ...
    mjSENS_RANGEFINDER: ClassVar[mjtSensor] = ...
    mjSENS_SUBTREEANGMOM: ClassVar[mjtSensor] = ...
    mjSENS_SUBTREECOM: ClassVar[mjtSensor] = ...
    mjSENS_SUBTREELINVEL: ClassVar[mjtSensor] = ...
    mjSENS_TENDONACTFRC: ClassVar[mjtSensor] = ...
    mjSENS_TENDONLIMITFRC: ClassVar[mjtSensor] = ...
    mjSENS_TENDONLIMITPOS: ClassVar[mjtSensor] = ...
    mjSENS_TENDONLIMITVEL: ClassVar[mjtSensor] = ...
    mjSENS_TENDONPOS: ClassVar[mjtSensor] = ...
    mjSENS_TENDONVEL: ClassVar[mjtSensor] = ...
    mjSENS_TORQUE: ClassVar[mjtSensor] = ...
    mjSENS_TOUCH: ClassVar[mjtSensor] = ...
    mjSENS_USER: ClassVar[mjtSensor] = ...
    mjSENS_VELOCIMETER: ClassVar[mjtSensor] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtSensor, value: int) -> None

        2. __init__(self: mujoco._enums.mjtSensor, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtSensor, value: int) -> None

        2. __init__(self: mujoco._enums.mjtSensor, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtSensor, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtSensor) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtSensor) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtSensor, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtSensor) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtSensor, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtSensor, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtSensor, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtSensor, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtSensor, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtSensor, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtSensor, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtSensor, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtSensor, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtSensor, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtSolver:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjSOL_CG: ClassVar[mjtSolver] = ...
    mjSOL_NEWTON: ClassVar[mjtSolver] = ...
    mjSOL_PGS: ClassVar[mjtSolver] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtSolver, value: int) -> None

        2. __init__(self: mujoco._enums.mjtSolver, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtSolver, value: int) -> None

        2. __init__(self: mujoco._enums.mjtSolver, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtSolver, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtSolver) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtSolver) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtSolver, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtSolver) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtSolver, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtSolver, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtSolver, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtSolver, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtSolver, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtSolver, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtSolver, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtSolver, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtSolver, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtSolver, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtStage:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjSTAGE_ACC: ClassVar[mjtStage] = ...
    mjSTAGE_NONE: ClassVar[mjtStage] = ...
    mjSTAGE_POS: ClassVar[mjtStage] = ...
    mjSTAGE_VEL: ClassVar[mjtStage] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtStage, value: int) -> None

        2. __init__(self: mujoco._enums.mjtStage, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtStage, value: int) -> None

        2. __init__(self: mujoco._enums.mjtStage, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtStage, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtStage) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtStage) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtStage, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtStage) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtStage, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtStage, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtStage, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtStage, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtStage, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtStage, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtStage, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtStage, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtStage, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtStage, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtState:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjNSTATE: ClassVar[mjtState] = ...
    mjSTATE_ACT: ClassVar[mjtState] = ...
    mjSTATE_CTRL: ClassVar[mjtState] = ...
    mjSTATE_EQ_ACTIVE: ClassVar[mjtState] = ...
    mjSTATE_FULLPHYSICS: ClassVar[mjtState] = ...
    mjSTATE_INTEGRATION: ClassVar[mjtState] = ...
    mjSTATE_MOCAP_POS: ClassVar[mjtState] = ...
    mjSTATE_MOCAP_QUAT: ClassVar[mjtState] = ...
    mjSTATE_PHYSICS: ClassVar[mjtState] = ...
    mjSTATE_PLUGIN: ClassVar[mjtState] = ...
    mjSTATE_QFRC_APPLIED: ClassVar[mjtState] = ...
    mjSTATE_QPOS: ClassVar[mjtState] = ...
    mjSTATE_QVEL: ClassVar[mjtState] = ...
    mjSTATE_TIME: ClassVar[mjtState] = ...
    mjSTATE_USER: ClassVar[mjtState] = ...
    mjSTATE_USERDATA: ClassVar[mjtState] = ...
    mjSTATE_WARMSTART: ClassVar[mjtState] = ...
    mjSTATE_XFRC_APPLIED: ClassVar[mjtState] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtState, value: int) -> None

        2. __init__(self: mujoco._enums.mjtState, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtState, value: int) -> None

        2. __init__(self: mujoco._enums.mjtState, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtState, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtState) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtState) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtState, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtState) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtState, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtState, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtState, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtState, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtState, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtState, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtState, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtState, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtState, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtState, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtStereo:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjSTEREO_NONE: ClassVar[mjtStereo] = ...
    mjSTEREO_QUADBUFFERED: ClassVar[mjtStereo] = ...
    mjSTEREO_SIDEBYSIDE: ClassVar[mjtStereo] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtStereo, value: int) -> None

        2. __init__(self: mujoco._enums.mjtStereo, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtStereo, value: int) -> None

        2. __init__(self: mujoco._enums.mjtStereo, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtStereo, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtStereo) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtStereo) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtStereo, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtStereo) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtStereo, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtStereo, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtStereo, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtStereo, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtStereo, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtStereo, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtStereo, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtStereo, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtStereo, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtStereo, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtTaskStatus:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjTASK_COMPLETED: ClassVar[mjtTaskStatus] = ...
    mjTASK_NEW: ClassVar[mjtTaskStatus] = ...
    mjTASK_QUEUED: ClassVar[mjtTaskStatus] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTaskStatus, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTaskStatus, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTaskStatus, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTaskStatus, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtTaskStatus) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtTaskStatus) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtTaskStatus) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtTaskStatus, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtTaskStatus, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtTexture:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjTEXTURE_2D: ClassVar[mjtTexture] = ...
    mjTEXTURE_CUBE: ClassVar[mjtTexture] = ...
    mjTEXTURE_SKYBOX: ClassVar[mjtTexture] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTexture, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTexture, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTexture, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTexture, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtTexture, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtTexture) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtTexture) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtTexture, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtTexture) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtTexture, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtTexture, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtTexture, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtTexture, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtTexture, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtTexture, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTexture, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTexture, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtTexture, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtTexture, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtTextureRole:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjNTEXROLE: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_EMISSIVE: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_METALLIC: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_NORMAL: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_OCCLUSION: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_OPACITY: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_ORM: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_RGB: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_RGBA: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_ROUGHNESS: ClassVar[mjtTextureRole] = ...
    mjTEXROLE_USER: ClassVar[mjtTextureRole] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTextureRole, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTextureRole, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTextureRole, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTextureRole, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtTextureRole) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtTextureRole) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtTextureRole) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtTextureRole, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtTextureRole, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtTimer:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjNTIMER: ClassVar[mjtTimer] = ...
    mjTIMER_ACTUATION: ClassVar[mjtTimer] = ...
    mjTIMER_ADVANCE: ClassVar[mjtTimer] = ...
    mjTIMER_COL_BROAD: ClassVar[mjtTimer] = ...
    mjTIMER_COL_NARROW: ClassVar[mjtTimer] = ...
    mjTIMER_CONSTRAINT: ClassVar[mjtTimer] = ...
    mjTIMER_FORWARD: ClassVar[mjtTimer] = ...
    mjTIMER_INVERSE: ClassVar[mjtTimer] = ...
    mjTIMER_POSITION: ClassVar[mjtTimer] = ...
    mjTIMER_POS_COLLISION: ClassVar[mjtTimer] = ...
    mjTIMER_POS_INERTIA: ClassVar[mjtTimer] = ...
    mjTIMER_POS_KINEMATICS: ClassVar[mjtTimer] = ...
    mjTIMER_POS_MAKE: ClassVar[mjtTimer] = ...
    mjTIMER_POS_PROJECT: ClassVar[mjtTimer] = ...
    mjTIMER_STEP: ClassVar[mjtTimer] = ...
    mjTIMER_VELOCITY: ClassVar[mjtTimer] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTimer, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTimer, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTimer, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTimer, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtTimer, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtTimer) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtTimer) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtTimer, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtTimer) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtTimer, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtTimer, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtTimer, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtTimer, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtTimer, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtTimer, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTimer, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTimer, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtTimer, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtTimer, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtTrn:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjTRN_BODY: ClassVar[mjtTrn] = ...
    mjTRN_JOINT: ClassVar[mjtTrn] = ...
    mjTRN_JOINTINPARENT: ClassVar[mjtTrn] = ...
    mjTRN_SITE: ClassVar[mjtTrn] = ...
    mjTRN_SLIDERCRANK: ClassVar[mjtTrn] = ...
    mjTRN_TENDON: ClassVar[mjtTrn] = ...
    mjTRN_UNDEFINED: ClassVar[mjtTrn] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTrn, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTrn, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtTrn, value: int) -> None

        2. __init__(self: mujoco._enums.mjtTrn, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtTrn, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtTrn) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtTrn) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtTrn, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtTrn) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtTrn, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtTrn, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtTrn, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtTrn, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtTrn, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtTrn, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtTrn, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtTrn, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtTrn, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtTrn, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtVisFlag:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjNVISFLAG: ClassVar[mjtVisFlag] = ...
    mjVIS_ACTIVATION: ClassVar[mjtVisFlag] = ...
    mjVIS_ACTUATOR: ClassVar[mjtVisFlag] = ...
    mjVIS_AUTOCONNECT: ClassVar[mjtVisFlag] = ...
    mjVIS_BODYBVH: ClassVar[mjtVisFlag] = ...
    mjVIS_CAMERA: ClassVar[mjtVisFlag] = ...
    mjVIS_COM: ClassVar[mjtVisFlag] = ...
    mjVIS_CONSTRAINT: ClassVar[mjtVisFlag] = ...
    mjVIS_CONTACTFORCE: ClassVar[mjtVisFlag] = ...
    mjVIS_CONTACTPOINT: ClassVar[mjtVisFlag] = ...
    mjVIS_CONTACTSPLIT: ClassVar[mjtVisFlag] = ...
    mjVIS_CONVEXHULL: ClassVar[mjtVisFlag] = ...
    mjVIS_FLEXBVH: ClassVar[mjtVisFlag] = ...
    mjVIS_FLEXEDGE: ClassVar[mjtVisFlag] = ...
    mjVIS_FLEXFACE: ClassVar[mjtVisFlag] = ...
    mjVIS_FLEXSKIN: ClassVar[mjtVisFlag] = ...
    mjVIS_FLEXVERT: ClassVar[mjtVisFlag] = ...
    mjVIS_INERTIA: ClassVar[mjtVisFlag] = ...
    mjVIS_ISLAND: ClassVar[mjtVisFlag] = ...
    mjVIS_JOINT: ClassVar[mjtVisFlag] = ...
    mjVIS_LIGHT: ClassVar[mjtVisFlag] = ...
    mjVIS_MESHBVH: ClassVar[mjtVisFlag] = ...
    mjVIS_PERTFORCE: ClassVar[mjtVisFlag] = ...
    mjVIS_PERTOBJ: ClassVar[mjtVisFlag] = ...
    mjVIS_RANGEFINDER: ClassVar[mjtVisFlag] = ...
    mjVIS_SCLINERTIA: ClassVar[mjtVisFlag] = ...
    mjVIS_SDFITER: ClassVar[mjtVisFlag] = ...
    mjVIS_SELECT: ClassVar[mjtVisFlag] = ...
    mjVIS_SKIN: ClassVar[mjtVisFlag] = ...
    mjVIS_STATIC: ClassVar[mjtVisFlag] = ...
    mjVIS_TENDON: ClassVar[mjtVisFlag] = ...
    mjVIS_TEXTURE: ClassVar[mjtVisFlag] = ...
    mjVIS_TRANSPARENT: ClassVar[mjtVisFlag] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtVisFlag, value: int) -> None

        2. __init__(self: mujoco._enums.mjtVisFlag, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtVisFlag, value: int) -> None

        2. __init__(self: mujoco._enums.mjtVisFlag, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtVisFlag) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtVisFlag) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtVisFlag) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtVisFlag, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtVisFlag, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtWarning:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjNWARNING: ClassVar[mjtWarning] = ...
    mjWARN_BADCTRL: ClassVar[mjtWarning] = ...
    mjWARN_BADQACC: ClassVar[mjtWarning] = ...
    mjWARN_BADQPOS: ClassVar[mjtWarning] = ...
    mjWARN_BADQVEL: ClassVar[mjtWarning] = ...
    mjWARN_CNSTRFULL: ClassVar[mjtWarning] = ...
    mjWARN_CONTACTFULL: ClassVar[mjtWarning] = ...
    mjWARN_INERTIA: ClassVar[mjtWarning] = ...
    mjWARN_VGEOMFULL: ClassVar[mjtWarning] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtWarning, value: int) -> None

        2. __init__(self: mujoco._enums.mjtWarning, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtWarning, value: int) -> None

        2. __init__(self: mujoco._enums.mjtWarning, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtWarning, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtWarning) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtWarning) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtWarning, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtWarning) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtWarning, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtWarning, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtWarning, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtWarning, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtWarning, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtWarning, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtWarning, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtWarning, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtWarning, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtWarning, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class mjtWrap:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    mjWRAP_CYLINDER: ClassVar[mjtWrap] = ...
    mjWRAP_JOINT: ClassVar[mjtWrap] = ...
    mjWRAP_NONE: ClassVar[mjtWrap] = ...
    mjWRAP_PULLEY: ClassVar[mjtWrap] = ...
    mjWRAP_SITE: ClassVar[mjtWrap] = ...
    mjWRAP_SPHERE: ClassVar[mjtWrap] = ...
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtWrap, value: int) -> None

        2. __init__(self: mujoco._enums.mjtWrap, value: int) -> None
        """
    @overload
    def __init__(self, value: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: mujoco._enums.mjtWrap, value: int) -> None

        2. __init__(self: mujoco._enums.mjtWrap, value: int) -> None
        """
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    @overload
    def __add__(self, arg0: int) -> int:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __add__(self, arg0: float) -> float:
        """__add__(*args, **kwargs)
        Overloaded function.

        1. __add__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __add__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    def __and__(self, arg0: int) -> int:
        """__and__(self: mujoco._enums.mjtWrap, arg0: int) -> int"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    @overload
    def __floordiv__(self, arg0: int) -> int:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __floordiv__(self, arg0: float) -> float:
        """__floordiv__(*args, **kwargs)
        Overloaded function.

        1. __floordiv__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __floordiv__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: mujoco._enums.mjtWrap) -> int"""
    def __int__(self) -> int:
        """__int__(self: mujoco._enums.mjtWrap) -> int"""
    def __lshift__(self, arg0: int) -> int:
        """__lshift__(self: mujoco._enums.mjtWrap, arg0: int) -> int"""
    @overload
    def __mod__(self, arg0: int) -> int:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __mod__(self, arg0: float) -> float:
        """__mod__(*args, **kwargs)
        Overloaded function.

        1. __mod__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __mod__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: int) -> int:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __mul__(self, arg0: float) -> float:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __mul__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    def __neg__(self) -> int:
        """__neg__(self: mujoco._enums.mjtWrap) -> int"""
    def __or__(self, arg0: int) -> int:
        """__or__(self: mujoco._enums.mjtWrap, arg0: int) -> int"""
    @overload
    def __radd__(self, arg0: int) -> int:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __radd__(self, arg0: float) -> float:
        """__radd__(*args, **kwargs)
        Overloaded function.

        1. __radd__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __radd__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    def __rand__(self, arg0: int) -> int:
        """__rand__(self: mujoco._enums.mjtWrap, arg0: int) -> int"""
    @overload
    def __rfloordiv__(self, arg0: int) -> int:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __rfloordiv__(self, arg0: float) -> float:
        """__rfloordiv__(*args, **kwargs)
        Overloaded function.

        1. __rfloordiv__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __rfloordiv__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: int) -> int:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __rmod__(self, arg0: float) -> float:
        """__rmod__(*args, **kwargs)
        Overloaded function.

        1. __rmod__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __rmod__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: int) -> int:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __rmul__(self, arg0: float) -> float:
        """__rmul__(*args, **kwargs)
        Overloaded function.

        1. __rmul__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __rmul__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    def __ror__(self, arg0: int) -> int:
        """__ror__(self: mujoco._enums.mjtWrap, arg0: int) -> int"""
    def __rshift__(self, arg0: int) -> int:
        """__rshift__(self: mujoco._enums.mjtWrap, arg0: int) -> int"""
    @overload
    def __rsub__(self, arg0: int) -> int:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __rsub__(self, arg0: float) -> float:
        """__rsub__(*args, **kwargs)
        Overloaded function.

        1. __rsub__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __rsub__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    def __rtruediv__(self, arg0: float) -> float:
        """__rtruediv__(self: mujoco._enums.mjtWrap, arg0: float) -> float"""
    def __rxor__(self, arg0: int) -> int:
        """__rxor__(self: mujoco._enums.mjtWrap, arg0: int) -> int"""
    @overload
    def __sub__(self, arg0: int) -> int:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    @overload
    def __sub__(self, arg0: float) -> float:
        """__sub__(*args, **kwargs)
        Overloaded function.

        1. __sub__(self: mujoco._enums.mjtWrap, arg0: int) -> int

        2. __sub__(self: mujoco._enums.mjtWrap, arg0: float) -> float
        """
    def __truediv__(self, arg0: float) -> float:
        """__truediv__(self: mujoco._enums.mjtWrap, arg0: float) -> float"""
    def __xor__(self, arg0: int) -> int:
        """__xor__(self: mujoco._enums.mjtWrap, arg0: int) -> int"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
