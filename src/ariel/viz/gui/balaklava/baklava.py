"""TODO(jmdm): description of script."""

from typing import Dict, Optional

from nicegui import ui


class Editor(
    ui.element,
    component="baklava.js",
):
    def __init__(self, options: Optional[Dict] = None) -> None:
        """Editor"""
        super().__init__()
        self._props["title"] = "title"
