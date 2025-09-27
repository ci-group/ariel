import numpy as _np
from _typeshed import Incomplete
from pathlib import Path as Path
from typing import Any as Any, Callable as Callable, TypeVar as TypeVar
from typing_extensions import Protocol

ArgsKwargs = tuple[tuple[Any, ...], dict[str, Any]]
ArrayLike = tuple[float, ...] | list[float] | _np.ndarray
PathLike = str | Path
FloatLoss = float
Loss = float | ArrayLike
BoundValue: Incomplete
X = TypeVar('X', covariant=True)

class JobLike(Protocol[X]):
    def done(self) -> bool: ...
    def result(self) -> X: ...

class ExecutorLike(Protocol):
    def submit(self, fn: Callable[..., X], *args: Any, **kwargs: Any) -> JobLike[X]: ...
