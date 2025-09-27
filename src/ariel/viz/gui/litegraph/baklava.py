"""TODO(jmdm): description of script."""

from typing import Callable, Dict, Optional

from nicegui import ui


class Editor(
    ui.element,
    component="baklava.js",
):
    def __init__(
        self, title: str, *, on_change: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self._props["title"] = title
        self.on("calculate", lambda e: print(e.args))
