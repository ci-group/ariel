import torch
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any

_TAGS: dict[str, dict[str, Any]]

class SupportLevel(Enum):
    """
    Indicates at what stage the feature
    used in the example is handled in export.
    """
    SUPPORTED = 1
    NOT_SUPPORTED_YET = 0
ArgsType = tuple[Any, ...]

def check_inputs_type(args, kwargs) -> None: ...
def _validate_tag(tag: str): ...

@dataclass(frozen=True)
class ExportCase:
    example_args: ArgsType
    description: str
    model: torch.nn.Module
    name: str
    example_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_args: ArgsType | None = ...
    tags: set[str] = field(default_factory=set)
    support_level: SupportLevel = ...
    dynamic_shapes: dict[str, Any] | None = ...
    def __post_init__(self) -> None: ...

_EXAMPLE_CASES: dict[str, ExportCase]
_MODULES: set[ModuleType]
_EXAMPLE_CONFLICT_CASES: dict[str, list[ExportCase]]
_EXAMPLE_REWRITE_CASES: dict[str, list[ExportCase]]

def register_db_case(case: ExportCase) -> None:
    """
    Registers a user provided ExportCase into example bank.
    """
def to_snake_case(name): ...
def _make_export_case(m, name, configs): ...
def export_case(**kwargs):
    """
    Decorator for registering a user provided case into example bank.
    """
def export_rewrite_case(**kwargs): ...
