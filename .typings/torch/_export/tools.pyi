import torch
from typing import Any

__all__ = ['report_exportability']

def report_exportability(mod: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None, *, strict: bool = True, pre_dispatch: bool = False) -> dict[str, Exception | None]:
    """
    Report exportability issues for a module in one-shot.

    Args:
        mod: root module.
        args: args to the root module.
        kwargs: kwargs to the root module.
    Returns:
        A dict that maps from submodule name to the exception that was raised when trying to export it.
        `None` means the module is exportable without issue.
    Sample output:
        {
            '': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_1': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_2': None
        }
    """
