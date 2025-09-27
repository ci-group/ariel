"""TODO(jmdm): description of script."""

# Standard library
from collections.abc import Callable
from pathlib import Path

# Third-party libraries
from nicegui import ui
from nicegui.element import Element
from rich.console import Console
from rich.traceback import install

# Local libraries
from signature_pad import SignaturePad

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
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Document</title>

    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@baklavajs/themes@2.0.2-beta.3/dist/syrup-dark.css" />
    <style>
        html,
        body {
            margin-left: 5vw;
            margin-right: 5vw;
            margin-top: 5vh;
            margin-bottom: 5vh;
        }

        #editor {
            width: 90vw;
            height: 90vh;
        }
    </style>
</head>
"""

body_html = """
<body>
    <div id="editor"></div>

    <script src="https://cdn.jsdelivr.net/npm/baklavajs@2.0.2-beta.3/dist/bundle.js"></script>
    <script>
        const viewModel = BaklavaJS.createBaklava(document.getElementById("editor"));
        const TestNode = BaklavaJS.Core.defineNode({
            type: "TestNode",
            inputs: {
                a: () => new BaklavaJS.RendererVue.TextInputInterface("Hello", "world"),
                c: () => new BaklavaJS.RendererVue.ButtonInterface("Name", () => console.log("Button clicked")),
            },
            outputs: {
                b: () => new BaklavaJS.RendererVue.TextInputInterface("Hello", "world"),
            },
        });
        viewModel.editor.registerNodeType(TestNode);
    </script>
</body>
"""


@ui.page("/")
def page() -> None:
    ui.add_head_html(head_html)
    ui.add_body_html(body_html)

    pad = SignaturePad().classes("border")
    ui.button("Clear", on_click=pad.clear)


if __name__ == "__main__":
    ui.run()
