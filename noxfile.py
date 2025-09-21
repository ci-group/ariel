"""Nox sessions.

Author:     jmdm
Date:       2025-09-13
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     Completed ✅

Notes
-----
    * https://github.com/wntrblm/nox
"""

# Standard library
import shutil
from pathlib import Path

# Third-party libraries
import nox

# --- NOX SETUP ---
nox.options.default_venv_backend = "uv"
package = "a2_base"
python_versions = ["3.12", "3.13"]


@nox.session(python=python_versions[0])
def docs(session: nox.Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    session.run(
        "uv",
        "sync",
        "--group",
        "docs",
        external=True,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    build_dir = Path("docs", "_build")

    if build_dir.exists():
        shutil.rmtree(build_dir)

    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.run("sphinx-autobuild", *args)
