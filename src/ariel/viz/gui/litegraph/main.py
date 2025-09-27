"""TODO(jmdm): description of script."""

from nicegui import ui
from nicegui.element import Element

head_html = """
<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/jagenjo/litegraph.js/css/litegraph.css">
<script type="text/javascript" src="https://cdn.jsdelivr.net/gh/jagenjo/litegraph.js/build/litegraph.min.js"></script>
"""
canvas = """<canvas id='node-editor' width='1024' height='720' style='border: 1px solid'></canvas>"""

from baklava import Editor


@ui.page("/")
def page() -> None:
    ui.add_head_html(head_html)
    editor = Editor(
        "State",
        on_change=lambda e: ui.notify(f"The value changed to {e.args}."),
    )

    with (
        ui.card()
        .classes("fixed_center")
        .style("style='width:100%; height:100%'")
    ):
        ui.html(canvas)


if __name__ == "__main__":
    ui.run()
