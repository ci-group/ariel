from ...ir import Buffer as Buffer, Layout as Layout, TemplateBuffer as TemplateBuffer
from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

_P = ParamSpec('_P')
_T = TypeVar('_T')

class ROCmTemplateBuffer(TemplateBuffer):
    workspace_size: Incomplete
    template: Incomplete
    def __init__(self, layout: Layout, inputs: Sequence[Buffer], make_kernel_render: Callable[_P, _T], workspace_size: int, template: ROCmTemplate) -> None: ...
    def get_workspace_size(self) -> int: ...
