import dataclasses
from .types import GuardFail as GuardFail, GuardFilterEntry as GuardFilterEntry
from torch._guards import GuardsSet as GuardsSet
from typing import Callable

@dataclasses.dataclass
class Hooks:
    guard_export_fn: Callable[[GuardsSet], None] | None = ...
    guard_fail_fn: Callable[[GuardFail], None] | None = ...
    guard_filter_fn: Callable[[list[GuardFilterEntry]], list[bool]] | None = ...
