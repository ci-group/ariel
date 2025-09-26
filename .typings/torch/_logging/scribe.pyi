from _typeshed import Incomplete
from typing import Callable
from typing_extensions import TypeAlias

TAtom: TypeAlias = int | float | bool | str
TField: TypeAlias = TAtom | list[TAtom]
TLazyField: TypeAlias = TField | Callable[[], TField]
open_source_signpost: Incomplete
