"""TODO(jmdm): description of script."""

# Standard library
from collections.abc import Callable
from pathlib import Path

# Third-party libraries
from nicegui import ui
from nicegui.element import Element
from rich.console import Console
from rich.traceback import install

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# Global functions
install(show_locals=True)
console = Console()

# Warning Control
# Type Checking
# Type Aliases

head_html = """
<head>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/jagenjo/litegraph.js/css/litegraph.css">
    <script type="text/javascript" src="https://cdn.jsdelivr.net/gh/jagenjo/litegraph.js/build/litegraph.min.js"></script>
</head>
"""
canvas = """<canvas id='node-editor' width='1024' height='720' style='border: 1px solid'></canvas>"""

html_code = """
var graph = new LGraph();
var canvas = new LGraphCanvas("#node-editor", graph);
graph.start()
"""

register_new_nodes = """
LiteGraph.wrapFunctionAsNode("ec/sum", sum, ["Number", "Number"], "Number")
"""


@ui.page("/")
def page() -> None:
    ui.add_head_html(head_html)

    with (
        ui.card()
        .classes("fixed_center")
        .style("style='width:100%; height:100%'")
    ):
        ui.html(canvas)
        ui.run_javascript(register_new_nodes)
        ui.run_javascript(html_code)

    data = {"name": "Bob", "age": 17}
    ui.number().bind_value(data, "age")
    ui.label().bind_text_from(data, "age", backward=lambda a: f"Age: {int(a)}")


if __name__ == "__main__":
    ui.run()
