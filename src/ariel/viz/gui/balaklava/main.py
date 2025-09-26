"""TODO(jmdm): description of script."""

from baklava import Editor
from nicegui import ui

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
    </script>
</body>
"""


@ui.page("/")
def page() -> None:
    ui.add_head_html(head_html)
    ui.add_body_html(body_html)
    editor = Editor()


if __name__ == "__main__":
    ui.run()
